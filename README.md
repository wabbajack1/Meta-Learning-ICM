# MAML

MAML (Model agnostic Meta learning) is comprised of two main components: the inner loop and the outer loop. The inner loop is responsible for learning a task, while the outer loop is responsible for learning the initial parameters of the model. We effectvally, optimizing gradient decent by gradient decent, by the outerloop. Here we differentiate through the inner loop optimizations! Remebering past optmization, which can be optimized (meta). The goal of MAML is to learn a set of initial parameters that can be quickly adapted to new tasks with a small number of examples. 

Short MAML: The **outer loop** updates the meta-initialization of the neural network parameters to a setting that enables fast adaptation to new tasks. The **inner loop** takes the outer loop initialization and performs task-specific adaptation over a few labeled samples. We are trying to optimize our few-shot optimization objective to find an optimal meta-parameter $\theta$, which we can easily fine-tune on any task with only a few respective samples [Luis M. et al.](https://interactive-maml.github.io/maml.html#start).

- [Paper MAML](https://arxiv.org/abs/1703.03400)
- [Youtube video MAML](https://www.youtube.com/watch?v=ItPEBdD6VMk)
- [Lectures; 3 vids](https://www.youtube.com/watch?v=h7qyQeXKxZE&list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A&index=64)
- [Libray learning2learn](https://github.com/learnables/learn2learn?tab=readme-ov-file)
- [Blog post MAML](https://interactive-maml.github.io/maml.html)
- [Blog 2 post MAML](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [Higher library video](https://www.youtube.com/watch?v=9XqP7zhYbMQ&t=62s)
- [What are fast weights?](https://syncedreview.com/2017/02/08/geoffrey-hinton-using-fast-weights-to-store-temporary-memories/)


## M-Way-K-Shot Learning

**M-way:** Die Anzahl der Klassen (oder Kategorien) in einer Aufgabe.

**K-shot:** Die Anzahl der Beispiele, die für jede Klasse in der Aufgabe bereitgestellt werden.

M-way K-shot Lernen umfasst Aufgaben, bei denen das Modell auf 
𝑀 Klassen mit 
𝐾 Beispielen pro Klasse als "Support Set" trainiert oder evaluiert wird.

## Komponenten

### Support Set

Enthält 
𝑀 × 𝐾 gekennzeichnete Beispiele.

Beispiel: Für eine 5-way 3-shot Aufgabe gibt es 
5 Klassen, jede mit 
3 gekennzeichneten Beispielen, was insgesamt 15 Beispiele ergibt.

Das Modell lernt die unterscheidenden Merkmale der Klassen aus diesem Set.

### Query Set

Enthält nicht gekennzeichnete Beispiele aus denselben 
𝑀 Klassen.

Die Aufgabe besteht darin, die Klasse jeder Abfrage basierend auf dem Support Set vorherzusagen.


## Experiments

- Pretraining and then fine-tuning on a new task (naive way)
- Vanilla MAML
  - Here show it in a K-shot learning setting, i.e. after the meta-learning phase, we are able to learn a new task with only K examples.
  - But show its limitations, i.e. it doesn't generalize well when encountering tasks in a seqeuential manner, i.e. continual learning.
- Meta-LSTM
- My approach: **FAME (Fisher Information and MAML with Elastic Weight Consolidation)**
  - Here the inner loop is augmented with Fisher Information Matrix, which is used to regularize the learning process.

## Ideas for new method

+ Integrate the ICM module:
  - Gradient Prediction: $L_{\text{ICM}} = \left\| \nabla_{\theta} L_{\text{CE}} (x, y; \theta) - \nabla_{\theta} \hat{L}_{\text{CE}} (x, y; \theta) \right\|^2$, where $\hat{L}$ is the predicted gradient.
  - Use $r_{curiousGradient}$, i.e. $r_{curiousGradient}= \left\| \nabla_{\theta} L_{\text{CE}} (x, y; \theta) - \nabla_{\theta} \hat{L}_{\text{CE}} (x, y; \theta) \right\|^2$, as a weighting factor in each sample of the classification loss. This reward encourages the model to focus on samples where the gradient pattern is novel or uncertain.
  - $L_{\text{weighted}} = \sum_{(x,y)} r_{\text{curiosity}} \cdot L_{\text{CE}} (x, y; \theta)$, where $r_{curiousGradient}$ curiosity is larger for samples with higher gradient novelty, effectively prioritizing learning from these samples. And hence gaining more information from them, i.e. from the data distribution, which might give more information about the task in a few-shot learning setting.
  - To stabilize the learning process, we have to use the Inverse model too, which helps us to learn only the most relevant features in the gradient feature space of the data distribution with respect to the gradients. understand "transitions"?

> Assumption of gradient direction:
> Gradient direction have much more information then steepest descent.

+ (optional idea) Efficiently Compute importance of parameters
Here we use **Memory Aware Synapses**. After completing each meta-learning step, calculate the importance of each parameter as it pertains to preserving previously learned tasks, maybe some Conjugate gradient?

### Algorithm
**Inner Loop:** During the task-specific adaptation phase, apply the ICM-based sample weighting to select data samples that produce the most informative gradients. This allows the model to adapt more effectively by focusing on novel patterns within each class.

**Outer Loop:** Use the accumulated insights from the ICM over multiple tasks to guide the meta-update. This helps the MAML model learn a meta-initialization that better generalizes across classes by prioritizing samples with informative gradients.

> Assumption:
> By using the ICM in this way, the model will prioritize learning from samples that contribute the most novel gradient information. This approach provides a strong foundation for enhancing meta-learning in classification by effectively utilizing **gradient-based curiosity**.