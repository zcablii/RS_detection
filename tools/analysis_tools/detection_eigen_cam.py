import os
from genericpath import isfile
import jittor as jt
from jittor import nn
from jittor import init
from tqdm import tqdm
import numpy as np
import jdet
import pickle
from jdet.config import get_cfg,save_cfg, init_cfg
from jdet.utils.visualization import visualize_results
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config import get_classes_by_name
from jdet.utils.general import build_file, current_time, sync,check_file,check_interval,parse_losses,search_ckpt
from jdet.data.devkits.data_merge import data_merge_result
from jdet.models.boxes.iou_calculator import BboxOverlaps2D_rotated_v1
from tqdm import tqdm
from jittor_utils import auto_diff
import copy
from typing import Callable, List, Tuple
import cv2
from PIL import Image
# jt.flags.lazy_execution=0
jt.cudnn.set_max_workspace_ratio(0.0)


init_cfg('/home/lyx/detection/JDet/configs/oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate.py') # oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate   orcnn_van3_for_test_1.py
class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,MODELS)
        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=self.model.named_parameters(), model=self.model)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)

        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        
        self.model_only = self.cfg.model_only
        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)
        print(self.resume_path)
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            if self.model_only:
                self.resume(model_only=True)
            else:
                self.resume(model_only=False)

    def get_modle(self):
        return self.model, self.optimizer,self.test_dataset
            
    @jt.no_grad()
    @jt.single_process_scope()
    def test(self):

        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            self.model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.test_dataset),total=len(self.test_dataset)):
                result = self.model(images,targets)
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])
                for mode in self.flip_test:
                    images_flip = images.copy()
                    if (mode == 'H'):
                        images_flip = images_flip[:, :, :, ::-1]
                    elif (mode == 'V'):
                        images_flip = images_flip[:, :, ::-1, :]
                    elif (mode == 'HV'):
                        images_flip = images_flip[:, :, ::-1, ::-1]
                    else:
                        assert(False)
                    result = self.model(images_flip,targets)
                    targets_ = copy.deepcopy(targets)
                    for i in range(len(targets_)):
                        targets_[i]["flip_mode"] = mode
                    results.extend([(r,t) for r,t in zip(sync(result),sync(targets_))])

            save_file = build_file(self.work_dir,f"test/test_{self.epoch}.pkl")
            pickle.dump(results,open(save_file,"wb"))
            if (self.cfg.dataset.test.type == "ImageDataset"):
                dataset_type = self.test_dataset.dataset_type
                data_merge_result(save_file,self.work_dir,self.epoch,self.cfg.name+'_epoch'+str(self.epoch),dataset_type,self.cfg.dataset.test.images_dir)
                # import sys
                # sys.path.append(os.path.join(os.getcwd(), "tools"))
                # from val import evaluate_in_training
                # evaluate_in_training(os.path.join(os.getcwd(),"submit_zips", self.work_dir.split(os.path.sep)[-1]+"_epoch"+str(self.epoch)+".csv"), self.iter, self.logger)

    def load(self, load_path, model_only=False):
        print('loading model')
        resume_data = jt.load(load_path)

        if ("model" in resume_data):
            self.model.load_parameters(resume_data["model"])
        elif ("state_dict" in resume_data):
            self.model.load_parameters(resume_data["state_dict"])
        else:
            self.model.load_parameters(resume_data)

        print(f"Loading model parameters from {load_path}")

    def resume(self, model_only=False):
        self.load(self.resume_path, model_only)

jt.flags.use_cuda = 1
runner = Runner()
model,optimizer,test_dataset = runner.get_modle()


fair_names = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 'Football_Field', \
              'Baseball_Field', 'Intersection', 'Roundabout', 'Bridge']

def predict(input_tensor, targets, model, detection_threshold):
    outputs = model(input_tensor,targets)
    pred_classes = [fair_names[i] for i in outputs[0][2].numpy()]
    pred_labels = outputs[0][2].numpy()
    pred_scores = outputs[0][1].detach().numpy()
    pred_bboxes = outputs[0][0].detach().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def poly2obb(polys):

    polys_np = polys.numpy()

    order = polys_np.shape[:-1]
    num_points = polys_np.shape[-1] // 2
    polys_np = polys_np.reshape(-1, num_points, 2)
    polys_np = polys_np.astype(np.float32)

    obboxes = []
    for poly in polys_np:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * np.pi
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)

    obboxes = obboxes.reshape(*order, 5)
    return jt.array(obboxes)

class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = jt.Var([0])
        

        if len(model_outputs[0]) == 0:
            return output

        self.bounding_boxes = poly2obb(self.bounding_boxes)
        for box, label in zip(self.bounding_boxes, self.labels):
            ious = BboxOverlaps2D_rotated_v1(box, model_outputs[0])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs[2][index] == label:
                score = ious[0, index] + model_outputs[1][index]
                output = output + score
        return output


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result
def show_cam(mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255


    cam =  heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def fasterrcnn_reshape_transform(x,targetSize = [128,128]):
    target_size = targetSize# x[-1].size()[-2:]
    activations = []
    for value in x:
        activations.append(
            nn.interpolate(
                jt.abs(value),
                target_size,
                mode='bilinear'))
    activations = jt.concat(activations, dim=1)
    return activations



class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform,cam_shape=[128,128]):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.cam_shape=cam_shape
        self.handles = []
        # print('target_layers',target_layers)
        print('target_layers length: ',len(target_layers))
        for target_layer in target_layers:

         
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
            self.handles.append(target_layer.register_forward_hook(self.save_activations))

    def save_activations(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation,targetSize = self.cam_shape)
        print('output: ', activation[0].shape)
        self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            print('direct return')
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad,targetSize = self.cam_shape)
            self.gradients = [grad.detach()] + self.gradients
        print('register_hook')
        output.register_hook(_store_grad)

    def __call__(self, x,input_targets):
        self.gradients = []
        self.activations = []
        return self.model(x,input_targets)

    def release(self):
        for handle in self.handles:
            handle.remove()

            
    
class BaseCAM():

    def __init__(self, model: nn.Module, target_layers: List[nn.Module], use_cuda: bool=False, reshape_transform: Callable=None,cam_shape=[128,128], compute_input_gradient: bool=False, uses_gradients: bool=True) -> None:
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        print('cam init')
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform,cam_shape)
        self.result0 = None
        self.result1 = None
    def get_cam_image(self, input_tensor: jt.Var, target_layer: nn.Module, targets: List[nn.Module], activations: jt.Var, grads: jt.Var, eigen_smooth: bool=False) -> np.ndarray:
        raise Exception('Not Implemented')

    def execute(self, input_tensor: jt.Var, input_targets, targets: List[nn.Module], eigen_smooth: bool=False) -> np.ndarray:
        print('cam exec')
        outputs = self.activations_and_grads(input_tensor,input_targets)
        if (targets is None):
            target_categories = np.argmax(outputs.numpy(), axis=(- 1))
            targets = [ClassifierOutputTarget(category) for category in target_categories]
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for (target, output) in zip(targets, outputs)])
            jt.grad(loss, input_tensor, retain_graph=True)
            # loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        self.result1 = cam_per_layer
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: jt.Var) -> Tuple[(int, int)]:
        (width, height) = (input_tensor.shape[(- 1)], input_tensor.shape[(- 2)])
        return (width, height)

    def compute_cam_per_layer(self, input_tensor: jt.Var, targets: List[nn.Module], eigen_smooth: bool) -> np.ndarray:
        assert len(self.activations_and_grads.activations) == 1
        activations_list = [a.numpy() for a in self.activations_and_grads.activations]
        print(activations_list)
        grads_list = [g.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if (i < len(activations_list)):
                layer_activations = activations_list[i]
            if (i < len(grads_list)):
                layer_grads = grads_list[i]
            print(layer_activations)
            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            self.result0 = cam
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def __call__(self, input_tensor: jt.Var, input_targets, targets: List[nn.Module]=None, aug_smooth: bool=False, eigen_smooth: bool=False) -> np.ndarray:
        return self.execute(input_tensor, input_targets, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f'An exception occurred in CAM with block: {exc_type}. Message: {exc_value}')
            return True

class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,cam_shape=[128,128]):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       cam_shape=[128,128],
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

def main():
    if not os.path.isdir('./img_cams'):
        os.mkdir('./img_cams')

    for batch_idx,(input_tensor,input_targets) in tqdm(enumerate(test_dataset),total=len(test_dataset)):
        print(input_targets[0]['img_file'])
        if batch_idx > 0:
            del targets
            del cam
        model.eval()
        boxes, classes, labels, indices = predict(input_tensor,input_targets, model, 0.9)
        target_layers = [model.neck]

        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
        print('in cam')
        cam = EigenCAM(model,
                    target_layers, 
                    use_cuda=True,
                    reshape_transform=fasterrcnn_reshape_transform,
                    cam_shape = [128,128])
                    # reshape_transform=None)

        grayscale_cam = cam(input_tensor,input_targets, targets=targets)
        # # Take the first image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam(grayscale_cam, use_rgb=True)
        # And lets draw the boxes again:
        # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
        cam_result = Image.fromarray(cam_image)
        cam_result.save('./img_cams/'+batch_idx+'.png')