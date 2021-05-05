# Generative Adversarial Networks (GANs)
**Neural Networks** have made tremendours Advancements in the last few years.They can now Recognise images ,Videos and Voice Messages with performance comparable to Humans.

But Even today ,Complete Automation without Human intervension is still  far fetched. Few tasks like those listed below still seem impossible to be done by a machine.
1. Predicting what is going to happen in the next Scenes of a Video or Making Paintings that dont really exist.
2. Writing an article online that will help many AI enthusiasts to get their hands dirty in this field . xD

Or maybe ? These difficult tasks can also be done!

**Generative Adversarial Networks**, or GANs in short, is an approach to generative modeling using deep learning methods, like CNN.

> The GAN architecture was first described in the 2014 paper by Ian Goodfellow, et al. titled “Generative Adversarial Networks.”

Generative modeling is an unsupervised learning task which involves automatically discovering and learning regularities and patterns in the input data in a way that the model can be used to generate new examples indistinguistable from the original data.

The main Ingrediants you will need to Develope a GAN for generating images are:
* **Discriminator** -> CNN model for classifying whether a given image is real or fake. 
* **Generator** -> Model that uses ICNN layers like Transpose Convolution to transform an noise input sampled from a ditribution into a 2D Image.

To have a clear Understanding about the working of GANs , let us take a real life example.

The Generator can be Interpreted as a forger who tries to make Fake Money.
The Discriminator can be Interpreted as a Police Man who tries to catch the Fake Currency.
So now its clear the forger will try its best to make the Fake money as close to real as possible and the Police will get better and better at distinguising between real and fake.
This is like a Game where both players want to outsmart their opponents.
After infinite time the forger will become so good that it will be able to generate Fake Money which has no difference to the real one and the police therefore will not be able to say with certainty if the money is real or fake . It will end up being a 50% probability that the money could be real or fake since it is indistinguishable.



![](https://i.imgur.com/LuTMHbX.jpg)

> Generative adversarial networks are based on a game theoretic scenario in which the generator network must compete against an adversary. The generator network directly produces samples. Its adversary, the discriminator network, attempts to distinguish between samples drawn from the training data and samples drawn from the generator. ~Goodfellow, et al.




## 1.Generator
The Generator Model takes a fixed length vector as Input.Values of the vector are Sampled Randomly from distributions like Gaussian.The vector is used to seed the generative process.

After training, points in this multidimensional vector space will correspond to points in the latent space (Zeta), forming a compressed representation of the data distribution. A latent space can be interpreted as a projection or compression of a data distribution.

In the case of GANs, the generator model applies meaning to points in a chosen latent space, such that new points drawn from the latent space can be provided to the generator model as input and used to generate new and different outputs.


## 2.Discriminator
The discriminator model takes an example from the domain as input (real or generated) and predicts a binary class label of real or fake.

The real example comes from the training dataset. The generated examples are output by the generator model.

The discriminator is just a binary classification model.

## Working 
Generator and Discriminator are trained together. The Generator generates a batch of samples.
Discrimator is fed these along with real example from the training Dataset for it to classify them as real or fake.

Backpropagation is done and the Discriminator is updated.
The generator is also updated based on how well, or not, the generated samples fooled the discriminator.

When the discriminator successfully identifies real and fake samples, it is rewarded or no change is needed to the model parameters, whereas the generator is penalized with large updates to model parameters.

Alternately, when the generator fools the discriminator, it is rewarded, or no change is needed to the model parameters, but the discriminator is penalized and its model parameters are updated.

At a limit, the generator generates perfect replicas from the Trainig Set every time, and the discriminator cannot tell the difference and predicts 50% for real and fake in every case. 

> Since the Training of Gans is quite unstable , Since it is not a simple optimisation problem,rather the two different models try to outsmart one another , there can be scenarios where we might not reach a Limit.

![](https://i.imgur.com/DTU4Lbo.png)


## Mathamatical Interpretation
> Let *x* be real data belonging to **X** (All Real Samples)
> Let *z* be the Latent Vector belonging to **zeta** (Latent Space)
> Let **G** represent the Generator, i.e **G** is a mapping from *z* to **G(*z*)**
> **G(*z*)** is the fake data
> Let **D** represent the Discriminator
> **D(*x*)** is Discriminators output on real Data
> **D(**G(*z*)**)** is Discriminators output on fake Data
> Let Error(a,b) be the error between a and b
> Let real image denote 1 and fake image denote 0

#### L_d (loss function of discriminator) will be
    L_d = Error(D(x),1)+Error(D(G(z)),0)

#### L_g (loss function of generator) will be
    L_g=Error(D(G(z)),1)

The Error function defined here is very generic and is just a function that tells us the distance or the difference between two functional parameter.

(In reality this Error function is something like Cross Entropy , KL divergence ,etc)
### Considering Binary Cross Entropy

    H(y,y') = − ∑ ylog(y') + (1 − y)log(1 − y')

#### Discriminator Loss
    L_d = − ∑ log(D(x)) + log(1 − D(G(z))) 
    where x belongs to X and z belongs to zeta
    Loss(D)=  max{log(D(x)) + log(1 − D(G(z)))}
    
<!-- ![](https://i.imgur.com/AjrFVkN.jpg)
 -->
 
#### Generator loss
    L_g = − ∑ log(D(G(z)) where z belongs to zeta
    Since log(1) = 0
    Loss(G) = min{log(D(x)) + log(1-D(G(z))} 
<!-- ![](https://i.imgur.com/LWdmAUw.jpg) -->

<!-- 
The original paper by Goodfellow presents a slightly different version of the loss functions shown above.

![](https://i.imgur.com/KyPWCLg.jpg) -  (B)

Essentially, the difference between (A) and (B) is the difference in sign, and whether we want to minimize or maximize a given quantity. In (A), we framed the function as a loss function to be minimized, whereas the original formulation presents it as a maximization problem, with the sign flipped.


Coming back to equation (B)  -->

#### Combined loss function
![](https://i.imgur.com/ZQh6W1P.jpg)

Remember that the above loss function is valid only for a single data point, to consider entire dataset we need to take the expectation of the above equation as

![](https://i.imgur.com/Xd2Pj0X.jpg)

which is the same equation as described in the original paper by Goodfellow et al.

Further More .

![](https://i.imgur.com/lI9ESqs.jpg)

![](https://i.imgur.com/kr9KNfY.jpg)
The goal of the discriminator is to maximize this value function. Through a partial derivative of V(G,D) with respect to D(x), we see that the optimal discriminator, denoted as D∗(x), occurs when

![](https://i.imgur.com/QJoWdaH.jpg)

Rearranging, we get

![](https://i.imgur.com/0sFuhPN.jpg)

And this is the condition for the optimal discriminator! Note that the formula makes intuitive sense: if some sample x is highly genuine, we would expect pdata(x) to be close to one and pg(x) to be converge to zero, in which case the optimal discriminator would assign 1 to that sample. On the other hand, for a generated sample x=G(z), we expect the optimal discriminator to assign a label of zero, since pdata(G(z)) should be close to zero.



