> **Note** 
> The SAE models used in this project were not developed from scratch by me. 
> They are extracted from an existing implementation available at: 
> [https://github.com/bartbussmann/BatchTopK](https://github.com/bartbussmann/BatchTopK). 
> The focus of this project is not the implementation and training of the SAEs, but the developement 
> of a method for interpreting the features obtained from them. 


There are a total of 4 different SAE implementations: 
- `VanillaSAE`: Original SAE implementation. 
- `JumpReLU`: SAE with `JumpReLU` activation function. 
- `BatchTopK`: SAE with `BatchTopK` activation function. 
- `BatchTopKJumpReLU`: SAE with `BatchTopK` for training and `JumpReLU` for inference. 