import os
import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import pickle
import copy
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from tabulate import tabulate
from transformers import AlbertTokenizer
from models.gmm import  save_gmm_audio, save_gmm_video,  train_gmm_audio, train_gmm_video
from utils.evaluations import eval_iemocap,eval_mosei
from utils.explainaibility import ExplainabilityFeatureProjector, SHAPExplainabilityWrapper
import shap


def save(toBeSaved, filename, mode='wb'):
    """
    Save an object to a file using pickle.

    Args:
        toBeSaved (object): The object to be saved.
        filename (str): The path to the file where the object will be saved.
        mode (str): The mode in which to open the file. Defaults to 'wb' (write binary).

    Returns:
        None
    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4)
    file.close()

class TrainerBase():
    """
    Base class for training models.

    Attributes:
        args (dict): Configuration arguments for training.
        model (torch.nn.Module): The model to be trained.
        best_model (dict): State dictionary of the best model.
        device (str): The device to run the model on (CPU or GPU).
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        dataloaders (dict): Dictionary containing data loaders for training and validation.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        earlyStop (int): Counter for early stopping.
        saving_path (str): Directory path for saving model and statistics.
    """

    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        self.args = args
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.earlyStop = args['early_stop']

        self.saving_path = f"./savings/"

    def make_stat(self, prev, curr): 
        """
        Generate a string representation of statistics.

        Args:
            prev (list): Previous statistics.
            curr (list): Current statistics.

        Returns:
            list: A list of strings indicating the change in statistics (↑, ↓, or unchanged).
        """
        new_stats = []
        for i in range(len(prev)):
            if curr[i] > prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2191')
            elif curr[i] < prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2193')
            else:
                new_stats.append(f'{curr[i]:.4f} -')
        return new_stats



    def save_stats(self, file_name:str):
        """
        Save training statistics and configuration to a file.

        Returns:
            None
        """
        stats = {
            'args': self.args,
            'train_stats': self.all_train_stats,
            'valid_stats': self.all_valid_stats,
            'test_stats': self.all_test_stats,
            'best_valid_stats': self.best_valid_stats,
            'best_epoch': self.best_epoch
        }

        save(stats, os.path.join(self.saving_path, 'stats', file_name))


    def save_model(self,modelname): 
        """
        Save the best-performing model's state dictionary to a file.

        Returns:
            None
        """

        model_path = os.path.join(self.saving_path, 'models', modelname)

        #  Ensure the directory exists before saving
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.save(self.best_model,  model_path)


class EmoTrainer(TrainerBase):
    """
    Trainer class for the Iemocap dataset.

    Inherits from TrainerBase and adds functionality specific to the Iemocap training process.
    """

    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(EmoTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.model=model
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
         
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []
        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['dataset']=='iemocap':
            self.eval_func = eval_iemocap 
        elif  self.args['dataset']=='mosei':  
            self.eval_func = eval_mosei

        if self.args['loss'] == 'bce' or 'focalloss':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]
            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]
        self.best_epoch = -1

    def train(self):
        """
        Execute the training process for a specified number of epochs.

        Returns:
            None

        """

        
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            stats,extracted_audio_features,extracted_visual_features  = self.train_one_epoch()
            if stats is None:
                print("Warning: train_one_epoch() returned None!")

            train_stats, train_thresholds = stats
            
            # Train GMM Every 3 Epochs
            if epoch % 1 == 0:  
               
               # Training and saving GMM 
               print(f"Training GMM after Epoch {epoch}")
               gmm_audio = train_gmm_audio(extracted_audio_features, num_components=4)
               save_gmm_audio(gmm_audio,epoch)  
               
               gmm_video = train_gmm_video(extracted_visual_features, num_components=4)
               save_gmm_video(gmm_video,epoch)  
               
               # Reload GMM after saving it
               self.model.A.reload_gmm_audio(epoch)
               self.model.reload_gmm_video(epoch)
            
           
            
            valid_stats, valid_thresholds = self.eval_one_epoch()
            
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)
            torch.save(self.model.state_dict(), "modelprime"+str(epoch)+".pth")

            
            for i in range(len(self.headers)):
                    for j in range(len(valid_stats[i])):
                        is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1))
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            self.best_valid_stats[i][j] = valid_stats[i][j]
                            if is_pivot:
                                self.earlyStop = self.args['early_stop']
                                self.best_epoch = epoch
                                self.best_model = copy.deepcopy(self.model.state_dict())
                        elif is_pivot:
                            self.earlyStop -= 1

                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))
        
        

        print('=== Best performance ===')
        for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats(self.args['model_name'])
        torch.save(self.model.state_dict(), self.args['model_name'])
        print('Results and model are saved!')

    def valid(self):
        """
        Validate the model on the validation set.

        Returns:
            None
        """
        # Reload GMM after saving it
        self.model.A.reload_gmm_audio()
        self.model.reload_gmm_video()
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[0][i]]], headers=self.headers[i]))
            print()

    def test(self):
    
        """
        Test the model on the test set.

        Returns:
            None
        """
        # Reload GMM after saving it
        self.model.A.reload_gmm_audio(6)
        self.model.reload_gmm_video(6)
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[0][i]]], headers=self.headers[i]))
            print()
        for stat in test_stats[0]:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()




    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            tuple: Training statistics and thresholds for evaluation.
        """
        self.model.train()
        if self.args['model'] == 'mme2e' and 'v' in self.args['modalities']:
            self.model.mtcnn.eval()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        all_extracted_audio_features = []
        all_extracted_visual_features = []
        

        pbar = tqdm(dataloader, desc='Train')
      
        for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in pbar:  
        
            
            
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

                   
            
            waveforms = waveforms.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            
            self.optimizer.zero_grad()
        
            with torch.set_grad_enabled(True):
               
                logits, class_audio_confidence_scores, class_visual_confidence_scores, audio_features,visual_features = self.model(imgs, imgLens, waveforms, text, return_features=True)
                all_extracted_audio_features.append(audio_features)
                all_extracted_visual_features.append(visual_features)
                
                loss = self.criterion(logits.squeeze(), Y, class_visual_confidence_scores,class_audio_confidence_scores)
               
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                    
                    
                
                loss.backward()

                self.optimizer.step()
                
            
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            print("train loss:{:.4f}".format(epoch_loss / data_size))

            # Free Memory
            del imgs, imgLens, waveforms, waveformLens, text, 
            torch.cuda.empty_cache()

            if self.scheduler is not None:
                self.scheduler.step()
        
       
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        epoch_loss /= len(dataloader.dataset)

     
        return self.eval_func(total_logits, total_Y),all_extracted_audio_features,all_extracted_visual_features

    def eval_one_epoch(self, phase='valid', thresholds=None):
        """
        Evaluate the model for one epoch.

        Args:
            phase (str): The phase to evaluate ('valid' or 'test').
            thresholds (list): Thresholds for evaluation metrics.

        Returns:
            tuple: Evaluation statistics and thresholds.
        """
        for m in self.model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        self.model.eval()

        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []

        
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in pbar:
        
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            

            waveforms = waveforms.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            
            with torch.set_grad_enabled(False):
               
                logits,class_audio_confidence_scores,class_visual_confidence_scores,_,_ = self.model(imgs, imgLens, waveforms,  text)
                loss = self.criterion(logits.squeeze(), Y, class_visual_confidence_scores,class_audio_confidence_scores)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
            
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            # Free Memory
            del imgs, imgLens, waveforms, waveformLens, text, 
            torch.cuda.empty_cache()

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        epoch_loss /= len(dataloader.dataset)
        
        
        
        return self.eval_func(total_logits, total_Y, thresholds)

