# WaveForge Thermal — focused prior-art and claim review

Дата поиска: `2026-08-29`.

## Scope

Это focused, а не systematic literature review. Рассмотрены первичные статьи
и официальные proceedings по learned warm-start topology optimization,
amortized/conditional design generation, neural reparameterization, direct
physics refinement и thermal topology optimization. Review выполнен после
teacher preflight NO-GO и не использовался для изменения success criteria.

## Наиболее близкие направления

| Work | Что уже сделано | Отношение к proposed WaveForge ML spike |
|---|---|---|
| Lin et al. (2018), *Investigation into the topology optimization for conductive heat transfer based on deep learning approach* ([DOI](https://doi.org/10.1016/j.icheatmasstransfer.2018.07.001)) | Deep fully convolutional/U-Net-like predictor заменяет позднюю часть SIMP optimization для conductive heat transfer после начальных iterations. | Уже покрывает neural acceleration thermal topology optimization; WaveForge не может заявлять первое применение U-Net или learned acceleration к теплопроводности. |
| Li et al. (2019), *Non-iterative structural topology optimization using deep learning* ([DOI](https://doi.org/10.1016/j.cad.2019.05.038)) | GAN получает thermal boundary conditions и генерирует low-resolution topology; SRGAN выполняет learned refinement. Training data получены множеством topology-optimization runs. | Уже покрывает amortized boundary-conditioned thermal design generation и показывает, что dataset-generation cost является центральной частью метода. |
| Hoyer, Sohl-Dickstein & Greydanus (2019), *Neural reparameterization improves structural optimization* ([paper](https://arxiv.org/abs/1909.04240)) | Density field параметризуется neural network, но weights оптимизируются для каждой задачи через physics objective. | Уже покрывает neural parameterization внутри exact optimization. Это не amortized warm-start, но запрещает claim, что само использование NN design variables является новым. |
| Nie et al. (2021), *TopologyGAN* ([paper](https://arxiv.org/abs/2003.04685), [DOI](https://doi.org/10.1115/1.4049533)) | Conditional generator использует physical fields, один раз вычисленные FEM на initial domain, и предсказывает topology для unseen boundary conditions. | Уже покрывает source/load/BC-conditioned design prediction с physics-derived channels. Source maps как network inputs сами по себе не являются новизной. |
| Deng et al. (2022), *Self-directed online machine learning for topology optimization* ([Nature Communications](https://www.nature.com/articles/s41467-021-27713-7)) | Online DNN surrogate и FEM объединены в optimization loop; среди demonstrations есть transient thermal energy-storage topology. | Уже покрывает online learned acceleration и thermal design с continuing exact simulations, хотя это surrogate/objective search, а не initializer prediction. |
| Chen, Joglekar & Kara (2024), *Topology Optimization Using Neural Networks With Conditioning Field Initialization for Improved Efficiency* ([DOI](https://doi.org/10.1115/1.4064131), [paper](https://arxiv.org/abs/2305.10460)) | Case-by-case neural topology optimization conditioning on initial strain-energy field reaches a target compliance in fewer iterations. | Уже формулирует iteration reduction from physics-conditioned initialization, хотя network не amortized across tasks. |
| Zhang et al. (2024), *Improving data-efficiency of deep generative model for fast design synthesis* ([DOI](https://doi.org/10.1007/s12206-024-0328-1), [primary PDF](https://web.mae.ufl.edu/nkim/Papers/paper146.pdf)) | DE-DGM обучается примерно на `100–200` designs, генерирует initial density guesses и затем запускает warm-start topology optimization. Для heat-conduction case reported warm-start cost около `38%` standard case. | Это наиболее прямой prior art: learned warm-start, downstream physics optimization, unseen conditions и thermal topology уже объединены. WaveForge не может заявлять саму комбинацию как новую. |
| Giannone et al. (NeurIPS 2023), *Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation* ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html), [paper](https://arxiv.org/abs/2305.18470)) | Conditional diffusion generation выравнивается с optimization trajectories; `5–10` direct optimization steps улучшают performance/manufacturability, включая OOD conditions. | Уже покрывает learned generation followed by direct physical refinement и OOD evaluation в structural topology optimization. |
| Nobari et al. (2024/2025), *NITO: Neural Implicit Fields for Resolution-free Topology Optimization* ([paper](https://arxiv.org/abs/2402.05073)) | Conditional neural implicit representation synthesizes topologies across domains/resolutions; accompanying evaluation включает direct optimization refinement of generated fields. | Уже покрывает amortized conditional design и physics refinement в более общей structural setting. |
| Li et al. (2024), *Machine-learning topology optimization with stochastic gradient descent optimizer for heat conduction problems* ([DOI](https://doi.org/10.1016/j.ijheatmasstransfer.2024.125226)) | Online-trained DNN predicts sensitivities inside heat-conduction topology optimization; reported total-time reductions достигают 70% при небольшом objective degradation. | Ещё один direct thermal-ML acceleration comparator; evaluation должна учитывать training/online-data cost и final objective degradation. |
| Zhang et al. (2025), *Intelligent Design Method for Thermal Conductivity Topology Based on a Deep Generative Network* ([article](https://link.springer.com/article/10.1186/s10033-025-01222-w)) | Conditional thermal design model uses heat-source distribution, volume fraction and heat-sink position; generated designs are compared with topology optimization and thermal behavior is experimentally checked. | Уже покрывает heat-source-conditioned thermal topology generation и independent physical checking in a broader applied setting. |

## Что WaveForge воспроизводит или планировал воспроизвести

- Mapping from problem-condition maps to an initial material field.
- Learned initialization followed by conventional differentiable physics
  optimization.
- Final checking by a numerical solver rather than accepting raw network output.
- Held-out boundary/source layouts and iteration-to-target metrics.
- Accounting for teacher/dataset cost instead of reporting inference-only
  speedup.

Все эти элементы имеют substantial prior art по отдельности, а несколько
работ уже объединяют generation, warm-start и direct refinement.

## Возможный узкий вклад — только после новых положительных данных

Потенциально различимой могла бы быть не общая ML architecture, а строгая
экспериментальная комбинация:

- один design для worst-case нескольких heat-source scenarios;
- exact material projection и strict-binary acceptance;
- independent high-resolution SciPy verification каждого refinement budget;
- registered perturbation robustness;
- comparison не только с random, но и с `MeanDesignInit` и
  `NearestNeighborInit`;
- complete break-even с teacher generation и training.

Даже эта комбинация должна проверяться более широким systematic search и
сравнением implementation details; она не является подтверждённой новизной.

## Запрещённые claims

WaveForge сейчас не может заявлять:

- first neural topology optimization;
- first learned warm-start for topology optimization;
- first thermal topology U-Net/CNN;
- first source-conditioned thermal design network;
- first learned design followed by physics refinement;
- first solver-verified or robustness-aware neural design;
- superiority to DE-DGM, DOM, NITO, TopologyGAN or thermal online-ML methods.

Локальное выполнение на RTX 4060 не является scientific novelty.

## Current evidence verdict

В WaveForge не обучалась neural network и не генерировался ML dataset: reduced
teacher failed preregistered fidelity gate (`11.0633%` median degradation при
limit `10%`). Поэтому текущая ветка не даёт положительного ML result и не
поддерживает claim о neural warm-start speedup, generalization или break-even.

Сильный результат проекта на этом этапе — validated physics и автоматический
solver-verified inverse design с небольшим (`~4.6–5.2%`) преимуществом над
prospectively optimized parametric Y-tree. Это inverse-design result, а не AI/ML
novelty result.
