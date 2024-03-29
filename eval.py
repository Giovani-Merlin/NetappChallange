import torch
import argparse
import numpy as np
from pprint import pprint
from sklearn import metrics

from dataloader import *
#from model import MRNet

def eval_image(image_path, model, threshold=0.5):
    image = np.load(image_path)
    image = np.stack((image,)*3, axis=1)

    image = torch.FloatTensor(image)
    if torch.cuda.is_available():
        image = image.cuda()
    #image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model.forward(image,[image.shape[0]])

    # 
    answer = torch.sigmoid(prediction).detach().cpu().numpy() > threshold
    prob = torch.sigmoid(prediction).detach().cpu().numpy()

    return answer, prob    

def eval_test(model,test_path, task,plane, threshold=0.5):
    validation_dataset = MRDataset(test_path, task, plane, train=False)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False,collate_fn=stack_collate)

    model.eval()
    model.eval()
    y_preds = []
    y_trues = []

    for i, (mris_batch,images_size, labels, weights) in enumerate(validation_loader):

        if torch.cuda.is_available():
            mris_batch = mris_batch.cuda()
            labels = labels.cuda()
            weights = weights.cuda()
        # Forward pass
        with torch.no_grad():
            prediction = model.forward(mris_batch,images_size)
        labels = labels.detach().cpu().numpy()
        y_trues.extend(labels)
        # 
        probas = torch.sigmoid(prediction).detach().cpu().numpy()
        y_preds.extend(probas)
    auc = metrics.roc_auc_score(y_trues, y_preds)
    y_preds = [y > threshold for y in y_preds]
    report = metrics.classification_report(y_trues, y_preds)
    confusion_metrics = metrics.confusion_matrix(y_trues, y_preds)

    print('AUC:', auc)
    pprint(report)
    pprint(confusion_metrics)
    return auc, report


def main(args, model):
    if args.image_path:
        eval_image(args.image_path, model,args.threshold)
    else:
        eval_test(model,args.test_path,args.task,args.plane, args.threshold)


if __name__ == '__main__':
    base_folder = os.getenv("BASE_FOLDER",".")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, 
                        choices=['abnormal', 'acl', 'meniscus'], default="abnormal")
    parser.add_argument('--plane', type=str,
                        choices=['sagittal', 'coronal', 'axial'], default='axial')
    parser.add_argument('--model_path', type=str, 
                        default="./models/model.pth")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--test_path", type=str, default=f"{base_folder}/data/")
    parser.add_argument('--image_path', type=str, help="As a numpy matrix")
    parser.add_argument('--threshold', type=int, default=0.2)
    args = parser.parse_args()
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    if torch.cuda.is_available():
        model = model.cuda()
    main(args, model)