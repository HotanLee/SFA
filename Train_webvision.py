from __future__ import print_function
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='PyTorch WebVision Parallel Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid1', default=0, type=int)
parser.add_argument('--gpuid2', default=1, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./webvision/', type=str, help='path to dataset')
parser.add_argument('--feat_size', default=1536, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '%s,%s'%(args.gpuid1,args.gpuid2)
random.seed(args.seed)
cuda1 = torch.device('cuda:0')
cuda2 = torch.device('cuda:1')

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, tmp_img_num_list, device,whichnet):
    criterion = SemiLoss()   
    
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device,non_blocking=True), inputs_x2.to(device,non_blocking=True), labels_x.to(device,non_blocking=True), w_x.to(device,non_blocking=True)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

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

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        feats, logits = net(mixed_input, return_features=True)
        logits2 = net.classify2(feats)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        loss_BCE = balanced_softmax_loss(mixed_target, logits2, tmp_img_num_list, "mean")
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty + loss_BCE
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\n')
        sys.stdout.write(' |%s Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                %(whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader,real_img_num_list,device,whichnet):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device,non_blocking=True) 
        optimizer.zero_grad()
        feats, outputs = net(inputs, return_features=True)
        outputs2 = net.classify2(feats)  

        labels = torch.zeros(labels.size(0), args.num_class).to(device).scatter_(1, labels.view(-1,1), 1)    
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * labels, dim=1))
        loss_BCE = balanced_softmax_loss(labels, outputs2, real_img_num_list, "mean")
        
        L = loss + loss_BCE

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\n')
        sys.stdout.write('|%s  Epoch [%3d/%3d] Iter[%4d/%4d]\t  Total-loss: %.4f CE-loss: %.4f  BCE-loss: %.4f'
                %(whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, L.item(), loss.item(), loss_BCE.item()))
        sys.stdout.flush()

        
def test(epoch,net1,net2,test_loader,device,queue):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    acc_meter.reset()
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device,non_blocking=True)
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)    
            outputs1 = net1.classify2(feats1)
            outputs2 = net2.classify2(feats2)

            outputs = outputs1 + outputs2 + outputs01 + outputs02
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    queue.put(accs)

def eval_train(epoch, eval_loader, model, num_all_img, cfeats_EMA, cfeats_sq_EMA, device, queue):    
    model.eval()
    total_features = torch.zeros((num_all_img, args.feat_size))
    total_labels = torch.zeros(num_all_img).long()
    confs_BS = torch.zeros(num_all_img)
    mask = torch.zeros(num_all_img).bool()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feats, outputs = model(inputs, return_features=True)
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]

            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
    
    total_features = total_features.to(device)
    total_labels = total_labels.to(device)

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i).nonzero(as_tuple=True)[0]
        idx_selected = (confs_BS[this_cls_idxs] > 0.02 * 1.005**epoch).nonzero(as_tuple=True)[0]
        idx_selected = this_cls_idxs[idx_selected]
        mask[idx_selected] = True

    refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)  # (10, 512)
    
    if epoch <= warm_up + 1:
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        cfeats_EMA = 0.9 * cfeats_EMA + 0.1 * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = 0.9 * cfeats_sq_EMA + 0.1 * refined_cfeats['cl2ncs'] ** 2
    
    # sample centers from gaussion postier
    sample_rate = 1
    refined_ncm_logits = torch.zeros((num_all_img, args.num_class)).to(device)
    ncm_classifier = KNNClassifier(args.feat_size, args.num_class)
    for i in range(sample_rate):
        mean = cfeats_EMA
        std = np.sqrt(np.clip(cfeats_sq_EMA - mean ** 2, 1e-30, 1e30))
        eps = np.random.normal(size=mean.shape)
        cfeats = mean + std * eps

        refined_cfeats['cl2ncs'] = cfeats
        ncm_classifier.update(refined_cfeats, device=device)
        refined_ncm_logits += ncm_classifier(total_features, None)[0]

    prob = get_gmm_prob(refined_ncm_logits, total_labels, device)
    queue.put(prob)

def get_gmm_prob(ncm_logits, total_labels, device):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # normalization, note that the logits are all negative
        this_cls_logits -=  np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)

        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).to(device)
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

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1))
    return loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def create_model(device):
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.to(device)
    return model

if __name__ == "__main__":
    
    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)    
    
    stats_log=open('./checkpoint/model_stats.txt','w') 
    test_log=open('./checkpoint/model_acc.txt','w')         
    
    warm_up = 1

    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_class = args.num_class,num_workers=8,root_dir=args.data_path,log=stats_log)
    print('| Building net')
    
    net1 = create_model(cuda1)
    net2 = create_model(cuda2)
    
    net1_clone = create_model(cuda2)
    net2_clone = create_model(cuda1)
    
    cudnn.benchmark = True
    
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.resume is not None:        
        resume_path = f'./checkpoint/model_webvision_e{args.resume}.pth'
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

    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
    warmup_trainloader1 = loader.run('warmup')
    warmup_trainloader2 = loader.run('warmup')
    real_img_num_list = torch.tensor(warmup_trainloader1.dataset.real_img_num_list)
    idx_class = warmup_trainloader1.dataset.idx_class
    num_all_img = torch.sum(real_img_num_list)
    print(max(real_img_num_list), min(real_img_num_list), max(real_img_num_list) / min(real_img_num_list))

    cfeats_EMA1 = np.zeros((args.num_class, args.feat_size))
    cfeats_sq_EMA1 = np.zeros((args.num_class, args.feat_size))
    cfeats_EMA2 = np.zeros((args.num_class, args.feat_size))
    cfeats_sq_EMA2 = np.zeros((args.num_class, args.feat_size))
    
    for epoch in range(start_epoch, args.num_epochs + 1):   
        lr=args.lr
        if epoch >= 50:
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr              

        if epoch <= warm_up:  
            p1 = mp.Process(target=warmup, args=(epoch,net1,optimizer1,warmup_trainloader1,real_img_num_list,cuda1,'net1'))                      
            p2 = mp.Process(target=warmup, args=(epoch,net2,optimizer2,warmup_trainloader2,real_img_num_list,cuda2,'net2'))
            p1.start() 
            p2.start()        

        else:        
            tmp_img_num_list1 = torch.zeros(args.num_class)
            pred1 = np.zeros(num_all_img, dtype=bool)
            tmp_img_num_list2 = torch.zeros(args.num_class)
            pred2 = np.zeros(num_all_img, dtype=bool)

            for i in range(args.num_class):
                pred1[idx_class[i]] = (prob1[idx_class[i]] > args.p_threshold)
                tmp_img_num_list1[i] = np.sum(pred1[idx_class[i]])

                pred2[idx_class[i]] = (prob2[idx_class[i]] > args.p_threshold)
                tmp_img_num_list2[i] = np.sum(pred2[idx_class[i]])

            labeled_trainloader1, unlabeled_trainloader1 = loader.run('train', pred2, prob2) # co-divide
            labeled_trainloader2, unlabeled_trainloader2 = loader.run('train', pred1, prob1) # co-divide
            
            p1 = mp.Process(target=train, args=(epoch,net1,net2_clone,optimizer1,labeled_trainloader1, unlabeled_trainloader1,tmp_img_num_list2,cuda1,'net1'))                             
            p2 = mp.Process(target=train, args=(epoch,net2,net1_clone,optimizer2,labeled_trainloader2, unlabeled_trainloader2,tmp_img_num_list1,cuda2,'net2'))
            p1.start()  
            p2.start()               

        p1.join()
        p2.join()
    
        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())
        
        q1 = mp.Queue()
        q2 = mp.Queue()
        p1 = mp.Process(target=test, args=(epoch,net1,net2_clone,web_valloader,cuda1,q1))                
        p2 = mp.Process(target=test, args=(epoch,net1_clone,net2,imagenet_valloader,cuda2,q2))
        
        p1.start()   
        p2.start()
        
        web_acc = q1.get()
        imagenet_acc = q2.get()
        
        p1.join()
        p2.join()        
        
        print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
        test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.flush()  
        
        eval_loader1 = loader.run('eval_train')          
        eval_loader2 = loader.run('eval_train')       
        q1 = mp.Queue()
        q2 = mp.Queue()
        p1 = mp.Process(target=eval_train, args=(epoch, eval_loader1, net1, num_all_img, cfeats_EMA1, cfeats_sq_EMA1, cuda1, q1))                
        p2 = mp.Process(target=eval_train, args=(epoch, eval_loader2, net2, num_all_img, cfeats_EMA2, cfeats_sq_EMA2, cuda2, q2))
        
        p1.start()   
        p2.start()
        
        prob1 = q1.get()
        prob2 = q2.get()

        p1.join()
        p2.join()

        if epoch % 10 == 0:
            save_path = f'./checkpoint/model_webvision_e{epoch}.pth'
            print(f'| Saving model to {save_path}')

            ckpt = {'net1': net1.state_dict(),
                    'net2': net2.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'prob1': prob1 if 'prob1' in dir() else None,
                    'prob2': prob2 if 'prob2' in dir() else None}
            torch.save(ckpt, save_path)
