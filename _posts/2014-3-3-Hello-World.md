---
layout: post
title: How Would Shakespeare Have Rapped? (A practical application of Deep Learning LSTMs)
---

# Deep-Rapspeare
Synthesising Rap-inflected Shakespeare using an LSTM Recurrent Neural Network inspired by Karpathy

## Introduction - Preparing to train — modifying code - Lo and behold! - Concluding Remarks  

*"MOTH. But do check it out your soul (May more) D on this boy."* - a brief snippet of the final result.

### Introduction
I was inspired by Andrej Karpathy's [post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), as well as seeing some other implementations of RNNs for text prediction to do something that combined both Shakespeare and the most popular current rappers.

### Preparing the training data

I downloaded the complete works of Shakespeare from [Project Gutenberg](http://www.gutenberg.org/ebooks/author/65), and got some scraped lyrics online. I here feel obliged to give a brief disclaimer: I'm not entirely clear on the legality of scraping metrolyrics' content, and, while I do know cases in the past where people have gotten in trouble (the [unfortunate Mr Swartz](https://www.wikiwand.com/en/United_States_v._Swartz) comes to mind), I don't think any harmful intention/ profit motives can be seriously argued, not to mention the fact that these lyrics sites don't even own the content they display.

  I haven't provided the code for scraping (As my father would always say, "You're either paranoid or you're stupid"), but it shouldn't be too big a challenge to flick through the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) documentation and/or look at some [existing implementations](https://github.com/mjbright/Scraper).
  I wrote a simple script (mixer.py) to alternate lines between the Shakespeare text and the Rap lyrics so that the RNN would be trained evenly on both texts, and because the sentence is the smallest unit of expression that preserves coherent syntax and ideas. If you had, say, alternated words between rap and Shakespeare, the result would be an uninteresting jumble of words with none of the distinct syntax that inflects either rap or Shakespeare.
  
### Modifying the existing Character RNN
  The character RNN I used was a Karpathy-inspired LSTM project [I found on Github](https://github.com/keskarnitish/Lasagne/blob/master/examples/lstm_text_generation.py) using Lasagne and Theano. In trying to to reuse this code, however, some things had to be changed. 
  
  First was the fact that this model didn't actually save any weights, so save_weights() was added, which lets you save the results of training to pass to your friends so they could have some great Rapspeare fun without an expensive GPU.
  
  Second, I reduced the number of total Epochs to 30 from 50, because I was noticing no real improvement after 20 or so Epochs (Like those kids that peak in high school, I saw that Rapspeare pretty much reached its potential under the design architecture that I'd chosen, so I decided to pull the plug there). Not to mention the fact that it was hurting my wallet a little bit under the p2 GPU server I was paying for per GB-hour. 
  
  I made some other smaller modifications (adding print() messages etc), but unfortunately did not get around to restructuring the whole thing. If you look at Rapspeare, you'll see that many of the functions are contained within a big *main(num_epochs=NUM_EPOCHS)* function. This isn't the best practice as it stops you from calling functions from outside the scope of *main()* something you might need to do in the python shell if you wanted to just generate text and not have the whole network retrain at the time.
  
  However, I was scared of toppling the precarious architecture around which I'd built RNN (It was a serious surprise/miracle to me that it worked at the time, in fact), and so sailing (I felt) between Charybdus and Scylla, I let the dormant beast that was Rapspeare lie, unperturbed (excuse the mongrel metaphors).
  
### Behold!
*"And Lo, for the Earth was empty of Form, and void. And Darkness was all over the Face of the Deep. And We said: 'Look at that fucker Dance." - David Foster Wallace*

  At last, after hours of brain-hurting debugging, reading documentation, and wondering why I was so stupid, a brief light appeared at the edge of my computer screen, and a voice said, *"Straight out the fuckin Dungeons of Rap, Pac put your boys get they were/ 'Bustin' like a pussy shooterstandin Flashflook, I am dressed Emilems."*
  The asyndeton in the first sentence, with 'Pac put your boys get they were', really evokes a sense of disorientation and hurry. You get the sense that the speaker is trying to escape from some abominable force and is perhaps senile (with Alzheimer's?). The portmanteau in the second line, 'shooterstandin', carries forth this feeling of unbelievable hurry, and, by my estimation, could be a poem in and of itself. Contemporary poets have done worse (see [Aram Samoyam](https://www.wikiwand.com/en/Aram_Saroyan) if you want to know what I mean).
  
  But this incredible feat of wordsmithing didn't come about suddenly. The first few iterations were a little less pretty. Note that the original seed phrase was 'The quick brown fox jumps'.
  
  After stopping it at less than a whole epoch , it was still in infant form, burbling in repetition: 
  
  `The quick brown fox jumps and the the the the the the the the the the the the the the the the the the the the...`
  
  
  After waiting for a little while (log loss = 1.60), things quickly got a little more interesting:
  
  `The quick brown fox jumpsmO
  One man Lord Herry try trouch,
  The voloun remaintry Chife, y'all know the copligee, I cun thou:
  That the coppitian this
  But the lose word discupis in can cause I what keetin' lipen on the bast of s`
  
 Here Rapspeare has gotten a pretty good ideaof capitilization, but is still consolidating its understanding of the patterns forming words, and so speaks largely in neologisms (if words like 'trouch' and 'coppitian' can be called such). Presumably the "y'all" comes from the rap side.
 
 I then grew tired of "the quick brown fox" and wanted to try something else as the seed text, so I used "Romeo, Romeo, wherefore art thou" to see if Rapspeare could figure out how to complete the phrase. Sure enough, I was entertained:
 
 `Romeo, Romeo, wherefore art thou Romeos; and so running too earshun
 To find so music her sisters, pretty bad good  do not meet mine yourselves to death...
 Canatic, and my Niggas...
 As much desires of the story
 Strike the tablest maid for your promises...
 Pale paradoles with rock, pardon: and 
 played open bottoms...
 O mama gatherates, I care another day youza hig.`
 
 Some parts are actually surprisingly poetic in a novel way; I quite like "To find so music" as an expression, and it works with "earshun". Also the alliteration of 'p' with "Pale paradoles with rock, pardon: and/ played..." Also, at this point you can glimpse the inflections of different voices (It feels as though a stiff, stumbling Tupac with lousy diction is summoned with the final line "O mama gatherates, I care another day youza hig") and the Rapspeare at this point can fill in at least the final word, albeit pluralised incorrectly, of the Shakespeare quote. This would have been around Epoch 7 (log loss function of about 0.9)
 
 And, after years of simulated rap battles, reading nothing but the finest English playwright and the most original contemporary rappers (graced by Kendrick Lamar, MF Doom, Nas, Tupac Shakur, Kanye West, Notorious B.I.G., and lesser rappers as the A$AP crew, Drake, Big Sean etc), Rapspeare comes out spitting fire: 
 
 **cue the music**
 
 `Straight out the fuckin dungeons of Rap, Pac put your boys get they were
 'Bustin' like a pussy shooterstandin Flashflook, I am dressed Emilems
 TOUCHSTONE. Now muthat's order man
 what has who incressed rap made saleThey say iesa
 MOTH. But do check it out your soul (May more) D on this boy
 Sire and fire her poremanion, God forgetter me
 When there's a perhnect down faith intents
 KING HENRY. Ay, so? Cut me, Master Saturn; and *let me get me raped* outBackin' law in voiceras are -a 
 Brok'r off! Nop`
 
 Drake's next ghostwriter?
 
 I found it quite curious that there was an inversion of the typical masculine bravado archetype with 'let me get me raped'. Is this some kind of subversive undercurrent underlying the machismo of Rap? Beneath the braggadocio, metal grills and chains, not only a type of penis-envy, but a desire to be sexually dominated? I'm just putting it out there.
 
### All's well that ends - Conclusion
As you can see, although the fact that Rapspeare worked was fascinating and amazing to me, there were clear limitations for the system, which was caused by decisions made in the design process. It couldn't really generate text that was cohesive past the level of a sentence, because I alternated both Rap and and Shakespeare in an effort to blend their tones.  

As you can see with Karpathy's example, if you use a complete longer text, you can generate text that is coherent over longer than merely a sentence, and that can actually maintain a store of symbols that are referenced over a paragraph, if the model is trained for long enough.
  
  Even for [examples](http://www.encore.ai) that I've seen that have been trained extensively on single artist, the neural net still lacks definite coherence, and I think that this will remain until some bigger breakthroughs occur in terms of some type of symbol storage, where the RNN can actually 'understand' the context, rather than this extremely elaborate pattern mapping.
  
  If anyone wanted to take a next step forward in creating the next rapper ghostwriter AI, it would be interesting to try and weight the ends of the sentence somehow such that it could follow a rhyme scheme, which in theory is definitely doable, but in practice might be difficult for someone without a lot of theano experience to muddle with the output process of the RNN.
  
  If you want to look through some more interesting output, I've uploaded a doc with some of the final training results (Lyrical_Rapspeare.txt).
  
I need to thank @sorenbouma for his significant help and guidance for this; I wouldn't have been able to do it without his encouragement, hints and generous time. 
  
