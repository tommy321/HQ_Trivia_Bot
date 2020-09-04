# HQ_Trivia_Bot
Neural Network Based Program to Cheat at HQTrivia

Now that HQTrivia is long dead, I figure I can post the Trivia Bot I obsessed over for a while. 

If you don't know, HQTrivia was a live trivia game on your phone where multiple choice answers were posted and you were given 10 seconds to select an answer. 
Wrong answers caused you to be kicked out of the game. Anyone still playing after a dozen or so questions would divide a pot of cash as a reward.

I only managed to win a couple of times, but making this program was a lot of fun and I learned a ton.

How it worked: 
- I cast my phone screen to my desktop computer. 
- The software watches the phone screen image, and uses tesseract to grab the trivia question, and this is parsed into question and answer text. 
- Various forms of the questions and each answer are sent to google and the result text page is parsed. 
- The question and the answers are also sent to wikipedia and the wikipedia article text is analyzed. 
- Word counts and cross correlation counts are pumped into a neural network that has been trained on a large bank of trivia questions. 
- The results are presented on the screen with a relative confidence for each answer. 
Generally it took 4-5 seconds to do the above, which left me with about 5 seconds to decide if I agreed with the computer. 

