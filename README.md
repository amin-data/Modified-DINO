# Modified Version of DINO
Pytorch code for modified version of DINO(Convit + modified loss)

This implementation uses Convit as the backbone of DINO and a modified loss function. Besides using the cls token as a signal, the modifed loss function also uses the patch embeddings of the teacher as another signal for the student network. 
