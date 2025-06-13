**Version Française. (SEE ENGLISH VERSION BELLOW)**
# AI _ Reinforcement


Ce projet m’a permis de débuter en apprentissage par renforcement à travers plusieurs environnements : Gridworld, Crawler et Pacman.   
À noter que je n’ai pas créé ce projet de toute pièce : il provient de notre cours de ML/IA. Ma contribution est l'implémentation des fichiers suivants :

valueIterationAgents.py : un agent utilisant l’itération sur les valeurs pour résoudre des MDP connus.  
qlearningAgents.py : des agents de Q-learning pour Gridworld, Crawler et Pacman.  
mdp.py : définit des méthodes générales pour les MDP.  
learningAgents.py : définit les classes de base ValueEstimationAgent et QLearningAgent, que vos agents vont étendre.  
 

Vous pouvez également consulter les fichiers de support suivants :

gridworld.py : implémentation de l’environnement Gridworld.  
featureExtractors.py : classes d’extraction de caractéristiques sur les paires (état, action), utilisées pour l’agent de Q-learning approximatif (dans qlearningAgents.py).

---

## PARTIE 1 _ Gridworld :

Ce projet mêle deux approches de l’apprentissage par renforcement :

Value Iteration (Itération sur les valeurs) :

Méthode déterministe et planificatrice.  
  - L’agent connaît à l’avance les règles de l’environnement (le MDP).  
  - Il calcule la meilleure stratégie en mettant à jour les valeurs de chaque état jusqu’à convergence.

Q-Learning :

Méthode apprenante et modèle-free.  
  - L’agent n’a pas connaissance des règles, mais apprend en interagissant avec l’environnement.  
  - Il ajuste ses estimations de la qualité d’une action dans un état (Q-valeurs) à partir des récompenses reçues.


nous commencerons par montrer la validité de nos agents sur Gridworld (PARTIE 1) Puis nous les appliquerons à un contrôleur de robot simulé (PARTIE 2 Crawler) et enfin à Pacman (PARTIE 3).


On définit d'abord une grille Gridworld qui a deux sorties, +1 et -1. Notre agents se déplace en utilisant les flèches du clavier. Notez que lorsque vous souhaitez aller vers le haut, l'agent ne monte que 80% du temps. 

Pour jouer :  
python gridworld.py -m

Un premier agent qui bouge aléatoirement :  
python gridworld.py -g MazeGrid

**Value Iteration**

J’ai implémenté l’agent `ValueIterationAgent` dans `valueIterationAgents.py`. Il applique une version batch de la value iteration pour calculer une politique optimale sur un MDP connu.

Méthodes implémentées :

* `computeQValueFromValues`  
* `computeActionFromValues`
* La commande suivante charge votre ValueIterationAgent, qui va calculer une politique et l’exécuter 10 fois. Appuyez sur une touche pour faire défiler les valeurs, les Q-valeurs et la simulation. Vous devriez constater que la valeur de l’état initial (V(start), affichée dans l’interface) et la récompense moyenne empirique (affichée après 10 exécutions) sont assez proches.

python gridworld.py -a value -i 100 -k 10



**Q-Learning** :

J’ai développé un agent Q-learning dans `qlearningAgents.py`. Il apprend une politique optimale par essais-erreurs sans connaître le modèle du MDP.

Méthodes implémentées :

* `getQValue`  
* `computeValueFromQValues`  
* `computeActionFromQValues`  
* `update`

Avec la mise à jour du Q-learning en place, vous pouvez observer votre agent apprendre manuellement, via le clavier :

python gridworld.py -a q -k 5 -m

Rappel : l’option -k contrôle le nombre d’épisodes d’apprentissage de votre agent. Observez comment il apprend sur l’état dans lequel il était, et non sur celui vers lequel il se déplace, "laissant l’apprentissage dans son sillage".

## PARTIE 2 - Crawler

J’ai mis en place une politique epsilon-greedy dans `getAction` pour équilibrer exploration et exploitation.  
This will invoke the crawling robot from class using our Q-learner. We can play around with the various learning parameters to see how they affect the agent's policies and actions. Note that the step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of our learning algorithm, and the discount factor is a property of the environment.

Notez également qu’il faut attendre environ 1000 étapes pour observer une réelle évolution du robot.

* python crawler.py

## PARTIE 3 - PACMAN

C’est l’heure de jouer à Pacman ! Pacman joue en deux phases. Durant la première phase, celle d’apprentissage, il commence à estimer la valeur des positions et des actions. Comme il faut beaucoup de temps pour apprendre des valeurs Q précises, même sur des petites grilles, les jeux d'entraînement de Pacman se déroulent sans interface graphique.

Une fois l’apprentissage terminé, Pacman entre en phase de test. À ce moment-là, self.epsilon et self.alpha sont fixés à 0.0, stoppant ainsi l’apprentissage et l’exploration, pour exploiter pleinement la politique apprise.

Test de pacman :  
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid   
Note : PacmanQAgent est identique, à l’exception de ses paramètres d’apprentissage par défaut, qui sont plus efficaces pour Pacman (epsilon = 0.05, alpha = 0.2, gamma = 0.8).


J’ai également implémenté ApproximateQAgent avec des fonctions de caractéristiques, ce qui permet une généralisation à des espaces d’états plus vastes.

Tests :  
* python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid  
À noter ici : Pacman ne prend pas encore en compte le fait que les fantômes ont peur de lui lorsqu’il a mangé une "power pellet"  
* python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic




# ENGLISH VERSION (Version Française dessus)
# AI _ Reinforcement

This project allowed me to get started with reinforcement learning through several environments: Gridworld, Crawler, and Pacman. Note i did not create this project from scratch and it is taken from our ML/IA course, my contribution reside in the files bellow :

* "valueIterationAgents.py": A value iteration agent for solving known MDPs.  
* "qlearningAgents.py": Q-learning agents for Gridworld, Crawler, and Pacman.  
* "mdp.py": Defines methods on general MDPs.  
* "learningAgents.py": Defines the base classes ValueEstimationAgent and QLearningAgent, which your agents will extend.  

  
  you can also see which are more support files :  
* "gridworld.py": The Gridworld implementation.  
* "featureExtractors.py": Classes for extracting features on (state, action) pairs. Used for the approximate Q-learning agent (in qlearningAgents.py).

---

## PART 1 - Gridworld

This project combines two approaches to reinforcement learning:

**Value Iteration** (deterministic and planning-based method):

* The agent knows the environment rules (the MDP) in advance.  
* It computes the optimal policy by updating the values of each state until convergence.

**Q-Learning** (learning-based and model-free method):

* The agent does not know the rules but learns by interacting with the environment.  
* It updates its estimates of the value of an action in a state (Q-values) based on received rewards.

We will start by validating our agents on Gridworld (PART 1), then apply them to a simulated robot controller (PART 2 - Crawler), and finally to Pacman (PART 3).

---


First, we define a Gridworld grid that has two exits: +1 and -1. Our agent moves using the arrow keys. Note that when you try to move up, the agent only succeeds 80% of the time.

To play:  
python gridworld.py -m

Run a first agent that moves randomly using the MazeGrid layout :   
python gridworld.py -g MazeGrid

### Value Iteration

I implemented the `ValueIterationAgent` in `valueIterationAgents.py`. It applies a batch version of value iteration to compute an optimal policy for a known MDP.

Implemented methods:

* computeQValueFromValues  
* computeActionFromValues

You can load your ValueIterationAgent, which will compute a policy and execute it 10 times. Press a key to cycle through values, Q-values, and the simulation. You should see that the value of the start state (V(start), displayed in the GUI) and the empirical average reward (printed after the 10 rounds) are quite close.

python gridworld.py -a value -i 100 -k 10

---

### Q-Learning

I developed a Q-learning agent in `qlearningAgents.py`. It learns an optimal policy through trial and error without knowing the MDP model.

Implemented methods:

* getQValue  
* computeValueFromQValues  
* computeActionFromQValues  
* update  

Once the Q-learning update is in place, you can watch your Q-learner learn under manual control, using the keyboard.

python gridworld.py -a q -k 5 -m

Reminder: the -k option controls the number of training episodes for your agent. Observe how it learns about the state it was in, not the one it moves to, "leaving learning in its wake."

---

## PART 2 - Crawler

I implemented an epsilon-greedy policy in `getAction` to balance exploration and exploitation.  
This invokes the crawling robot from class using our Q-learner. You can play around with various learning parameters to see how they affect the agent's policies and actions. The step delay is a parameter of the simulation, while the learning rate and epsilon are parameters of the learning algorithm. The discount factor is a property of the environment.

Note that you'll need to wait until around 1000 steps to see meaningful evolution in the crawler's behavior.

* python crawler.py

---

## PART 3 - Pacman

Time to play some Pacman!  
Pacman plays in two phases. In the first (training), he learns the values of positions and actions. Because learning accurate Q-values takes a long time, training runs in quiet mode by default (no GUI or console display). In the second (testing), Pacman’s epsilon and alpha are set to 0.0, stopping learning and exploration to let him exploit his learned policy.

You can test the game with PacmanQAgent, which has default learning parameters suited for the Pacman environment (epsilon=0.05, alpha=0.2, gamma=0.8).

* train for 2000 episodes, test for 10 on smallGrid.  
* python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 

---

I also implemented the `ApproximateQAgent` using feature functions, which allows for generalization to larger state spaces.

* Run ApproximateQAgent on smallGrid with 2000 training episodes and 10 tests.  
* python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid


  Note that Pacman does not yet account for ghosts being scared after eating a power pellet.

* Run ApproximateQAgent with the SimpleExtractor on the mediumClassic layout with 50 training and 10 testing episodes.  
* python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

---
