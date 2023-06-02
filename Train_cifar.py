from __future__ import print_function
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--noise_mode',  default='imb')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise ratio')
parser.add_argument('--arch', default='resnet18', type=str, help='resnet18')
parser.add_argument('-w', '--warm_up', default=30, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)
parser.add_argument('--beta', default=0.99, type=float, help='smoothing factor')
parser.add_argument('--phi', default=1.005, type=float, help='parameter for dynamic threshold')
parser.add_argument('--sample_rate', default=1, type=int, help='sampling rate of SFA')
args = parser.parse_args()

def set_seed():
    torch.cuda.set_device(args.gpuid)
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed()

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,current_labels, tmp_img_num_list):
    net.train()
    net2.eval() #fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):    
        labels_x = current_labels[index_x]
        try:
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)  
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)  
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        feats, logits = net(mixed_input, return_features=True)
        logits2 = net.classify2(feats)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        logits2_x = logits2[:batch_size*2]
        logits2_u = logits2[batch_size*2:]
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        loss_BCE_x, loss_BCE_u = balanced_softmax_loss_Semi(logits2_x, mixed_target[:batch_size*2], logits2_u, mixed_target[batch_size*2:], tmp_img_num_list)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + loss_BCE_x + lamb * loss_BCE_u + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        feats, logits1 = net(inputs, return_features=True)
        loss = CEloss(logits1, labels)

        logits2 = net.classify2(feats)
        loss_BCE = balanced_softmax_loss(labels, logits2, torch.tensor(dataloader.dataset.real_img_num_list), "mean")

        L = loss + loss_BCE
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)           
            outputs1 = net1.classify2(feats1)
            outputs2 = net2.classify2(feats2)
            outputs = outputs1 + outputs2 + outputs01 + outputs02
            _, predicted = torch.max(outputs, 1)            

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 

            total_preds.append(predicted)
            total_targets.append(targets)

    acc = 100.*correct/total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_class)]

    print("\n| Test Epoch #%d\t Accuracy: %.2f%% %s\n" %(epoch,acc, str(cls_acc)))
    test_log.write('Epoch:%d   Accuracy:%.2f %s\n'%(epoch,acc, str(cls_acc)))
    test_log.flush()  

def eval_train(model, cfeats_EMA, cfeats_sq_EMA):    
    model.eval()
    total_features = torch.zeros((num_all_img, feat_size))  # save sample features
    total_labels = torch.zeros(num_all_img).long()  # save sample labels
    tmp_img_num_list = torch.zeros(args.num_class)  # compute N_k from clean sample set
    pred = np.zeros(num_all_img, dtype=bool)  # clean probability
    confs_BS = torch.zeros(num_all_img)  # confidence from ABC
    mask = torch.zeros(num_all_img).bool()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, outputs = model(inputs, return_features=True)
            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
            
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]
    
    total_features = total_features.cuda()
    total_labels = total_labels.cuda()

    # Instant Centroid Estimation
    tau = 1 / args.num_class
    sampled_class = []  # classes that need to do SFA 
    for i in range(args.num_class):
        idx_selected = (confs_BS[idx_class[i]] > tau * args.phi ** epoch).nonzero(as_tuple=True)[0]
        idx_selected = idx_class[i][idx_selected]
        mask[idx_selected] = True
        if (idx_selected.size(0) > 300):
            sampled_class.append(i)
        
    remained_class = list(set(range(args.num_class)) - set(sampled_class))
    refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)  # (10, 512)
    
    # stochastic feature averaging
    if epoch == warm_up + 1:
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        cfeats_EMA = args.beta * cfeats_EMA + (1 - args.beta) * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = args.beta * cfeats_sq_EMA + (1 - args.beta) * refined_cfeats['cl2ncs'] ** 2
    
    # sample centers from gaussion postier
    refined_ncm_logits = torch.zeros((num_all_img, args.num_class)).cuda()
    for i in range(args.sample_rate):
        mean = cfeats_EMA
        std = np.sqrt(np.clip(cfeats_sq_EMA - mean ** 2, 1e-30, 1e30))
        eps = np.random.normal(size=mean.shape)
        cfeats = mean + std * eps

        refined_cfeats['cl2ncs'][sampled_class] = cfeats[sampled_class]
        refined_cfeats['cl2ncs'][remained_class] = mean[remained_class]

        ncm_classifier.update(refined_cfeats, device=args.gpuid)
        refined_ncm_logits += ncm_classifier(total_features, None)[0]

    prob = get_gmm_prob(refined_ncm_logits, total_labels)

    for i in range(args.num_class):
        pred[idx_class[i]] = (prob[idx_class[i]] > args.p_threshold)
        tmp_img_num_list[i] = np.sum(pred[idx_class[i]])

    return pred, prob, tmp_img_num_list

def get_gmm_prob(ncm_logits, total_labels):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()

def get_knncentroids(feats=None, labels=None, mask=None):

    feats = feats.cpu().numpy()
    labels = labels.cpu().numpy()

    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype('bool')
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()
            
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_==i) & mask_], axis=0))
        return np.stack(centroids)

    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels, mask)
    
    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,   
            'cl2ncs': cl2n_centers}

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def balanced_softmax_loss_Semi(logits_x, targets_x, logits_u, targets_u, sample_per_class):
    spc = sample_per_class.type_as(logits_x)

    spc_x = spc.unsqueeze(0).expand(logits_x.shape[0], -1)
    logits_x = logits_x + spc_x.log()
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    spc_u = spc.unsqueeze(0).expand(logits_u.shape[0], -1)
    logits_u = logits_u + spc_u.log()
    probs_u = torch.softmax(logits_u, dim=1)
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu

if args.warm_up is not None:
    warm_up = args.warm_up
else:
    warm_up = 30

store_name = '_'.join([args.dataset, args.arch, args.imb_type, str(args.imb_factor), args.noise_mode, str(args.noise_ratio)])

log_name = store_name + f'_w{warm_up}'
if args.resume is not None:
    log_name += f'_r{args.resume}'

if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
stats_log = open('./checkpoint/%s'%(log_name)+'_stats.txt','w') 
test_log = open('./checkpoint/%s'%(log_name)+'_acc.txt','w')

loader = dataloader.cifar_dataloader(args.dataset, imb_type=args.imb_type, imb_factor=args.imb_factor, noise_mode=args.noise_mode, noise_ratio=args.noise_ratio,\
    batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log)
args.num_class = 100 if args.dataset == 'cifar100' else 10
feat_size = 512
ncm_classifier = KNNClassifier(feat_size, args.num_class)
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()
warmup_trainloader = loader.run('warmup')
num_all_img = len(warmup_trainloader.dataset)
idx_class = []  # index of sample in each class
for i in range(args.num_class):
    idx_class.append((torch.tensor(warmup_trainloader.dataset.noise_label) == i).nonzero(as_tuple=True)[0])
cfeats_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_EMA2 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA2 = np.zeros((args.num_class, feat_size))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

model_name = store_name
if args.resume is not None:
    if args.resume > warm_up:
        model_name += f'_w{warm_up}'
    
    resume_path = f'./checkpoint/model_{model_name}_e{args.resume}.pth'
    print(f'| Loading model from {resume_path}')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        prob1 = ckpt['prob1']
        prob2 = ckpt['prob2']
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        model_name = store_name
        start_epoch = 1
else:
    start_epoch = 1

CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()

for epoch in range(start_epoch, args.num_epochs + 1):   
    lr = args.lr
    if epoch > 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch <= warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:
        current_labels1 = noisy_labels
        current_labels2 = noisy_labels

        pred1, prob1, tmp_img_num_list1 = eval_train(net1, cfeats_EMA1, cfeats_sq_EMA1)  # sample selection from net1
        pred2, prob2, tmp_img_num_list2 = eval_train(net2, cfeats_EMA2, cfeats_sq_EMA2)  # sample selection from net2

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, current_labels2, tmp_img_num_list2) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, current_labels1, tmp_img_num_list1) # train net2      

    test(epoch,net1,net2)

    if epoch in [warm_up]:
        save_path = f'./checkpoint/model_{model_name}_e{epoch}.pth'
        print(f'| Saving model to {save_path}')

        ckpt = {'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'prob1': prob1 if 'prob1' in dir() else None,
                'prob2': prob2 if 'prob2' in dir() else None}
        torch.save(ckpt, save_path)

    if epoch == warm_up:
        model_name += f'_w{warm_up}'
