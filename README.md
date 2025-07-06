Quantum Faster R-CNN Implementation in Pytorch
========

This repository implements [Faster R-CNN](https://arxiv.org/abs/1506.01497) with training, inference and map evaluation in PyTorch, along with integration of quantum circuits in the RoI Head using PennyLane. The implementation of the base classical Faster R-CNN model is attributed to ExplainingAI's YouTube video [here](https://youtu.be/Qq1yfWDdj5Y?si=_iOYLiBPmBPz4uTv).
The aim was to create a simple implementation based on PyTorch faster r-cnn codebase and to integrate it with quantum computing principles.

The implementation caters to batch size of 1 only and uses roi pooling on single scale feature map.
The repo is meant to train faster r-cnn on voc dataset.



# Quickstart
* Create a new conda environment with python 3.11.10 then run below commands
* ```cd FasterRCNN-PyTorch```
* ```pip install -r requirements.txt```
* For quickstarting, run the notebook `quantum-faster-rcnn-on-voc-2007.ipynb` or `variant-<name>.ipynb` in the root directory of the repo.

## Data preparation
For setting up the VOC 2007 dataset:
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007` folder
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007-test` folder
* Place both the directories inside the root folder of repo according to below structure
    ```
    FasterRCNN-Pytorch
        -> VOC2007
            -> JPEGImages
            -> Annotations
        -> VOC2007-test
            -> JPEGImages
            -> Annotations
        -> tools
            -> train.py
            -> infer.py
            -> train_torchvision_frcnn.py
            -> infer_torchvision_frcnn.py
        -> config
            -> voc.yaml
        -> model
            -> faster_rcnn.py
        -> dataset
            -> voc.py
    ```

## For training on your own dataset

* Copy the VOC config(`config/voc.yaml`) and update the [dataset_params](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/config/voc.yaml#L1) and change the [task_name](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/config/voc.yaml#L35) as well as [ckpt_name](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/config/voc.yaml#L41) based on your own dataset.
* Copy the VOC dataset(`dataset/voc.py`) class and make following changes:
   * Update the classes list [here](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/dataset/voc.py#L61) (excluding background).
   * Modify the [load_images_and_anns](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/dataset/voc.py#L13) method to returns a list of im_infos for all images, where each im_info is a dictionary with following keys:
     ```        
      im_info : {
		'filename' : <image path>
		'detections' : 
			[
				'label': <integer class label for this detection>, # assuming the same order as classes list present above, with background as zero index.
				'bbox' : list of x1,y1,x2,y2 for the bboxes.
			]
	    }
     ```
* Ensure that `__getitem__` returns the following:
  ```
  im_tensor(C x H x W) , 
  target{
        'bboxes': Number of Gts x 4,
        'labels': Number of Gts,
        }
  file_path(just used for debugging)
  ```
* Change the training script to use your dataset [here](https://github.com/AlifioDitya/Quantum-Faster-RCNN/blob/main/tools/train_torchvision_frcnn.py#L41)
* Then run training with the desired config passed as argument.

## Classical Differences from Faster RCNN paper
This repo has some differences from actual Faster RCNN paper.
* Caters to single batch size
* Uses a randomly initialized fc6 fc7 layer of 1024 dim.
* Most of the hyper-parameters have directly been picked from official version and have not been tuned to this setting of 1024 dimensional fc layers. As of now using this, model achieves ~61-62% mAP.
* To improve the results one can try the following:
  * Use VGG fc6 and fc7 layers
  * Tune the weight of different losses
  * Experiment with roi batch size
  * Experiment with hard negative mining

## For modifications 
* To change the fc dimension , change `fc_inner_dim` in config
* To use a different backbone, make the change [here](https://github.com/explainingai-code/FasterRCNN-PyTorch/blob/main/model/faster_rcnn.py#L748) and also change `backbone_out_channels` in config
* To use hard negative mining change `roi_low_bg_iou` to say 0.1(this will ignore proposals with < 0.1 iou)
* To use gradient accumulation change `acc_steps` in config to > 1

## Main Modifications from Classical Faster R-CNN
* Addition of Andrea Mari's DQC (2020)
```
n_qubits = config["quantum_params"]["n_qubits"]
q_depth = config["quantum_params"]["q_depth"]
q_delta = config["quantum_params"]["q_delta"]

# Define the quantum device
dev = qml.device('default.qubit', wires=n_qubits)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)

weight_shapes = {"weights": (q_depth, n_qubits)}

def H_layer(n_qubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def RY_layer(weights):
    """Layer of parametrized qubit rotations around the y-axis."""
    for idx, weight in enumerate(weights):
        qml.RY(weight, wires=idx)

def entangling_layer(n_qubits):
    """Layer of entangling gates (CNOTs)."""
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)

# Custom layer to integrate quantum processing into a PyTorch model
class DressedQuantumNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DressedQuantumNet, self).__init__()
        self.pre_net = nn.Linear(input_dim, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, output_dim)
        
    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        
        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        
        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = torch.hstack(quantum_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        
        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)
```
* Integration of DQC wrapper in quantum heads
```
# Use quantum layers for classification and bounding box regression
if (config['model_params']['quantum_head']):
    self.cls_layer = DressedQuantumNet(input_dim=self.fc_inner_dim, output_dim=num_classes)
    self.cls_layer = self.cls_layer.to(device)
    self.bbox_reg_layer = DressedQuantumNet(input_dim=self.fc_inner_dim, output_dim=num_classes * 4)
    self.bbox_reg_layer = self.bbox_reg_layer.to(device)
else:
    self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
    self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
    
    torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
    torch.nn.init.constant_(self.cls_layer.bias, 0)

    torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
    torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)
``` 
You can disable the quantum integration by setting the `quantum_head` integration to `False`.

## Using torchvision FasterRCNN 
* For training/inference using torchvision faster rcnn codebase, use the below commands passing the desired configuration file as the config argument.
* ```python -m tools.train_torchvision_frcnn``` for training using torchvision pretrained Faster R-CNN class on voc dataset
   * This uses the following arguments other than config file
   * --use_resnet50_fpn
      * True(default) - Use pretrained torchvision faster rcnn
      * False - Build your own custom model using torchvision faster rcnn class)
* ```python -m tools.infer_torchvision_frcnn``` for inference and testing purposes. Pass the desired configuration file as the config argument.
   * This uses the following arguments other than config file
   * --use_resnet50_fpn
      * True(default) - Use pretrained torchvision faster rcnn
      * False - Build your own custom model using torchvision faster rcnn class)
      * Should be same value as used during training
   * --evaluate (Whether to evaluate mAP on test dataset or not, default value is False)
   * -- infer_samples (Whether to generate predicitons on some sample test images, default value is True)

## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of faster r-cnn on voc dataset  

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of FasterRCNN the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples/*.png```

## Citations
Mari, A., Bromley, T. R., Izaac, J., Schuld, M., & Killoran, N. (2020). Transfer learning in hybrid classical-quantum neural networks. Quantum, 4, 340
Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1137–1149.
