This file describes several iterations of the spectrogram classification model that have been developed over the 2 month internship period and my estimation of their effectiveness.

Version 1: Autoencoder with resonances
Initially the autoencoder would only reconstruct black images. My guess is that since the majority of pixels were black, it was playing it safe and guessing the images were all black. To combat this, false negative were given a larger penalty than false positives. This fixed the issue.
The model was a resounding sucesses in that it had very high auc values and created clearly distinct classes based on where the stationary resonances occured. It proved effective for multiple window sizes and it was this model along with the bokeh plot that was presented in the first meeting. The only downside was that stationary resonances already well understood and really the only thing that the model understood outside of the resonances was that there were these noisy images that sometimes contained donward sliding gliders; it seems that the model was just detecting high activity however.

Note: After the first model was trained and working, time was taken to clean up code and verify optimal parameters though as a rule of thumb the parameters were good enough or they were catastrophically awful.

Then: During the code review it was suggested to train the autoencoder with: the sam model with resonances; the sam model without resonances; the normal model with resonances.

Version 2: Autoencoder with Sam and resonances
The autoencoder was effective in reproducing the binary spectrograms but the tsne clustering yeilded no physical separation. Given inferior performance even with the resonances as well as comparatively long runtimes, the sam model was abandoned and never tested without resonances.

Version 3: Autoencoder without resonances
Shortly after the Sam model was abandoned, the original model was tried with the resonances removed. With the bokeh plot I was able to verify the sucessful removal of resonances and at a glance the clustering seemed to be working. This would turn out to be deceptive; the model was actually just learning the geometry of the image as well as the activity level and separating this way.

Version 4: Autoencoder without resonances with rotation/ smaller windows
In order prevent the model from learning the geometry of the image two methods were attempted: First the window size was decreased which though I beleive the actual reconstruction still worked, did nothing to solve the problem. Next, the window was increased to 200 seconds (to create a square window) and all four orientations were fed into the autoencoder as the input chanels; the target was the correct orientation. Unfortunately, the rotations completely prevented the autoencoder from reconstructing the images. Several loss biases were tried but the image would always turn out all black or all white. No intermediate threshold was found that fixed the issue and the rotation method was abandoned.

Version 5: Manual Slope Convolution Model Without Resonances
As a last ditch effort to get something working without the resonances, I tried replacing the autoencoder with a series of manually selected slope convolution filters and row wise pooling and frequency binning. Clustering this manually created latent space produced great results and classes that are usually pretty easy to justify. During this process, hyperparameters were tweaked in a less than systematic way untill the desired results were arrived upon.

Version 6: Manual Slope Convolution Model Without Resonances + Autoencoder
To increase the efficacy of the newly created model, the manual latent spaces were to be fed into an autoencoder to tease out additional patterns. In order to make the latent spaces large enough to be fed through yet another autoencoder, the hyper parameters were adjusted away from what optimal for the previous model version. Several attempts have been made to cluster using the new model with dubious effectiveness. Even the bokeh plots for this method are not amazing. However, it is very possible the poor efficacy is due to the limited amount of time that has been spent playing around with the model.
