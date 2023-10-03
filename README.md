# GPTmodel
## Goal

The primary objective of this project is to deepen our understanding of the Transformers' architecture as described in the ["Attention is all you need" ](https://arxiv.org/abs/1706.03762) paper. At first , I followed the[ video by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=0Qw0nNYas0c-iY_l) to implement a GPT-like model that, based on Shakespearean writing, could generate new coherent text. To ensure comfort and better interpretability of the results in my native language, French, and for the sake of enjoyment, I switched the training corpus to the French version of Harry Potter 7.

## Experimental Steps
You can find details of these steps in the experimental notebooks.

### Bigram Model
Initially, I attempted to implement a simple bigram model, which predicted the most probable character based on the preceding characters. The results were predictably not very satisfactory, but this approach helped me grasp the fundamental workings of the model.

### GPT Model at Character Level
Subsequently, still following the video, I enhanced the model by implementing multihead attention, ultimately achieving a model closely resembling the decoder described in the 2017 paper.
 
### GPT Model at Word Level
Driven by curiosity to explore word-level predictions rather than character-level predictions, I made slight adjustments to the code to obtain a GPT model that predicts word by word.

## How to Use

To utilize the model, simply launch the main notebook, which is a cleaned version of the GPT model at the word level. The notebook includes monitoring of the loss using the Weights and Biases library, as well as saving of the weights during training.

You can customize all the hyperparameters according to your needs and then execute the code. Feel free to modify the training corpus. Also, remember to set the word_level boolean variable to the appropriate value based on whether you want to train your model at the character level or the word level.



## result 

This is a sample of the best result I could have by training the model for less than an hour (between 30min and 1 hours), this is not perfect but we can imagine that by scaling up the model (augmenting the number of head, of context, and training for longuer) we could obtain really satisfying result

Here is an example of the optimal output obtained from training the model for a relatively short duration, typically between 30 minutes and 1 hour. Although it is not flawless, we can imagine achieving considerably better results by upscaling the model (augmenting the number of heads, expanding the context, and extending the training duration etc..). 

### At character level 

Harry parut envie s’effondré sur les Mangemorts de Noël et restèrent dans leurs inombreux. Mrs Cattermole ne connut dessus
pas comment si peut-être la campe, chercha Xenophilius.

— S’il passa une coude disparure en l’intervit.

— Non, et plus, les Reliques d’autres : de l’Ordre qui allait presque ici. Ils se généra en deux murs, n’est pas
trembler, il se poussa à nouveau et s’était vapabili, elle essaya d’un bon sombre et glissa un hall perles de
Dobby.

— Tu ne sais étendu un nouveau clan. Un front aux recheveux ? Ils se croisaient à reconnuer une doute qui
l’apparaîtrise. Vous n’avait-elle pas attaqué Harry, ni j’étais très bien de quelques papieds.

— Mais dans celle connaissait de sa lescrule obscurité en dormir, et ensemble, moi transformé les baguettes mots
qui ont gêné sûr, je peux un peux désir !


### At word level 

— Hermione, répondit Ron. Je suis tous les voyageurs eux à revenir par-dessus de Beedle la
baguette ! Vous lui avez préparé la fenêtre de l’héritage.

Je crois qu’on va non ? cria Ron.

— Attention, je ne crois pas, précisa Hermione. Maintenant quelque part, devenu à présent silencieusement du
ennemi à moins, on aurait dit : je ne peux pas vous travailler avec lui qui en annonça une
fille. Si ce qu’il a de suite – le premier d’au sujet. envoyait en tout intéressant que Bill sembla
partis par vous avez récupéré, le demanda-t-on, je le moyen de Harry aux ailes de ses bagages
authentique des années qu’il dit :

— Hermione est des regards. Nous devons être leur recette, éclairé une minute à présent devant
des années. Il transplana dans cette distance, j’ai été dit : il s’est soulagé. Alors qu’on faut subir le monde d’origine avec lui…

