# debug memories lmao

this folder holds all the history of us fixing that huge architecture failure where the edge branch mode-collapsed into predicting all zeroes. 

we had to go super deep into the weeds:
- switching out standard bce for tversky loss
- blindly slamming the pos_weight to 7766 and making the model sweat 
- running the dense text hallucination checks on authentic docs
- writing pilot scripts and testing the bayar noise branch to see if augmentations destroyed the signal

it was basically non stop sanity checks, overwriting files, and almost breaking the model before getting it right lmao. keeping this here as a fun memory of solving it.
