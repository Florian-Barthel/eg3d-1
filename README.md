## 3D Multiview Inversion

<a href="florian-barthel.github.io/multiview-inversion/">Project page</a>

### Prepare Video Data

Coming soon

### Run Inversion

Multi-view Inversion
```bash
python multiview_inversion.py --network=path/to/pkl --target=path/to/data --num-steps=500 --num-steps_pti=500 --outdir=./out --num-targets=7
```

Multi-latent Inversion
````bash
python multiview_inversion_multi_w.py --network=path/to/pkl --target=path/to/data --num-steps=500 --num-steps_pti=500 --outdir=./out --num-targets=7 --continue-w=path/to/checkpoint --use-interpolation=True --depth-reg=True --w-norm-reg=True
````


## Citation

Multiview Inversion
```
@misc{barthel2023multiview,
      title={Multi-view Inversion for 3D-aware Generative Adversarial Networks}, 
      author={Florian Barthel and Anna Hilsmann and Peter Eisert},
      year={2023},
      eprint={2312.05330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

EG3D
```
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
```

## Acknowledgements

This work has partly been funded by the German Research Foundation (project 3DIL, grant
no. 502864329) and the German Federal Ministry
of Education and Research (project VoluProf, grant
no. 16SV8705).
