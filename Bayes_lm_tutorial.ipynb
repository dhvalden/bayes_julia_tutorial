{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "environmental-evidence",
   "metadata": {},
   "source": [
    "# Bayesian linear regression from scratch in Julia"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "boolean-clarity",
   "metadata": {},
   "source": [
    "In this notebook I will attempt to recreate [this](https://zjost.github.io/bayesian-linear-regression/) tutorial of  Bayesian linear regression using Julia language, but with a prose and examples hopefully more relatable to a social scientist. I am helping myself by referring to [this](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2885293/) paper and to the [doctoral thesis](https://ora.ox.ac.uk/objects/uuid:bf6c3fb5-5208-4dfe-aa0a-6e6da45c0d87) of one of my excellent supervisors Richard P. Mann. This exercise has three main objectives. The first one is procrastination. If I'm going to waste my time I may as well waste it in some productive way. The second is to increase my limited understanting of Bayesian inference. I can always read some more manuals and explanations, but without some hands-on activity I will never be able to fully get it. Also, I want to share and refine this tutorial with my fellow social scientist, so we can make the knowledge of mathematical concepts more accesible, instead of keeping them under the custody of a reclusive band of nerds (myself included). And third, I want to learn Julia language. Julia is a relatively new programming language with a clear focus on scientific computing, with the added feature that can be compiled for performance. I'm getting fed up of Tidyverse users proclaiming their gospel, and Python 'data scientist' trying to apply deep learning to everything, and Julia looks like a nice escape from that."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "pressing-cancer",
   "metadata": {},
   "source": [
    "Since this is a tutorial of a fairly complex mathematical algorithm, the use of mathematical explressions will be inevitable. I'm not a mathematician, and probably I don't understand these concepts enough, but I will try my best to explain them clearly. Hopefully this effort will help me to understand them better with the added bonus of communicate them to you, the reader."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "pharmaceutical-radiation",
   "metadata": {},
   "source": [
    "## The Bayes law and conditional probabilities"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dutch-atlas",
   "metadata": {},
   "source": [
    "The Bayes law is a probability theorem, so we migth benefit from reviewing some core concepts and terminology."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "considerable-story",
   "metadata": {},
   "source": [
    "The probability of an event A is commonly denoted $P(A)$ and is usually defined as the number of desired outcomes, divided by the total number of all outcomes (e.g. number of heads on the total number of coin tosses)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cultural-trinidad",
   "metadata": {},
   "source": [
    "Conditional probability of an event A given B is denoted $P(A \\mid B)$ and is defined as the probability of A occuring given that B already occurred, and can be calculated as follows:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "numerical-findings",
   "metadata": {},
   "source": [
    "$$P(A \\mid B) = \\frac{P(A \\cap B)}{P(B)} \\tag{1}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "individual-mechanics",
   "metadata": {},
   "source": [
    "The above equation can also be rearranged as follows:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "reflected-bermuda",
   "metadata": {},
   "source": [
    "$$P(A \\cap B) = P(A \\mid B)P(B) \\tag{2}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "least-lighting",
   "metadata": {},
   "source": [
    "Where $P(A \\cap B)$ is the probability that both events A and B occur. We must note that $P(A \\mid B)$ is not equal to $P(B \\mid A)$."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "unable-tonight",
   "metadata": {},
   "source": [
    "However, we do know that:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "portuguese-islam",
   "metadata": {},
   "source": [
    "$$P(A \\cap B) = P(B \\cap A) \\tag{3}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "common-surface",
   "metadata": {},
   "source": [
    "Meaning that the joint propability of A and B is the same as the joint probability of B and A. Now, taken equation (2) and (3) into consideration we can also say that:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cardiovascular-antique",
   "metadata": {},
   "source": [
    "$$P(A \\mid B)P(B) = P(B \\mid A)P(A) \\tag{4}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fifty-ribbon",
   "metadata": {},
   "source": [
    "And if we rearrange equation (4) to leave only one term in th left side we arrive to the Bayes law, which describes the relationship between $P(A \\mid B)$ and $P(B \\mid A)$."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "helpful-healthcare",
   "metadata": {},
   "source": [
    "$$P(A \\mid B) = \\frac{P(B \\mid A)P(A)}{P(B)} \\tag{5}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "peripheral-prerequisite",
   "metadata": {},
   "source": [
    "## Bayes in linear regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "exterior-benjamin",
   "metadata": {},
   "source": [
    "### Setting up the stage"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "textile-balloon",
   "metadata": {},
   "source": [
    "Now, lets try to use the Bayes law, described in equation (5), to device a way to estimate the parameters of a simple linear regression. Lets first recall the parts and the general form of a simple linear regression:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "enhanced-conversation",
   "metadata": {},
   "source": [
    "$$y=\\beta_{0}+\\beta_{1}X+\\epsilon \\tag{6}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "relevant-project",
   "metadata": {},
   "source": [
    "In equation (6), $y$ is generally the outcome variable we are trying to model. Thier values can be known, and we are using its data to crate a model to predict future values of the variable, or we simply want to estimate the model parameters as a way to discribe its relationship with $X$. $X$ is the variable we used to describe $y$ and is generally called the predictor. the set of parameters $\\{\\beta_{0}, \\beta_{1}, \\epsilon\\}$ are commonly called the intercept, the slope and the error, respectively. For simplicity, these parameters can be gruped into regression coeficients (intercept and slope), and noise (or error). These are unknown, and we hope to estimate them from the data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "novel-expression",
   "metadata": {},
   "source": [
    "From a traditional frequentist stand point, the goal of the estimation process is to get a \"point estimate\". This means to get a single value for each parameter. Now, from a Bayesian stand point this is not acceptable, because we don't want only the probability of obtaining a single particular value. We want a way to estimate the probability of obtaining *any* value the parameters can take (any value in the parameters space) giving the information (data) available. So, in our notation, we will move from the expression $P(A)$ denoting the probability of a discrete event, to $p(a)$ (lowercase) denoting the probability density function of a continious random variable, describing its probability distribution."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "trained-greenhouse",
   "metadata": {},
   "source": [
    "Let $\\theta$ denote any parameter we wish to estimate (e.g. any parameter of our linear regression) and let $data$ be the data we have available for that purpose. Then, using the Bayes law, the general way of obtaining the probability distribution for a parameter given the data available, is described by:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "rocky-louis",
   "metadata": {},
   "source": [
    "$$p(\\theta \\mid data) = \\frac{p(data \\mid \\theta)p(\\theta)}{p(data)} \\tag{7}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "descending-causing",
   "metadata": {},
   "source": [
    "In Bayesian jargon, the terms composing equation (7) have particular names. $p(\\theta \\mid data)$ is known as the *posterior* distribution, as it represents the distribution of $\\theta$ *posterior* to observing (*given* that we have observed) the $data$. $p(\\theta)$ is known as the *prior* distribution of the parameter, and it represents any prior belief or information we could have about $\\theta$ (e.g. previous studies, general information, etc). If we don't have any prior information, there is methods of incorporating a *prior* with minimal information. $p(data \\mid \\theta)$ is known as the *likelihood* and can also be written $\\mathcal{L}(\\theta \\mid data)$. In traditional frequentist estimation, the likelihood is used to estimate the unknown parameters of a given model, by maximising the likelihood function (whatever is the function in each particular case). This is known as *Maximum Likelihood Estimation (MLE)*. Maximising the likelihood function means obtaining the optimum set of parameters so it makes the observed data the most probable given our model. From a Bayesian point of view, this is not acceptable, because the goal of the Bayesian estimation is to find the most probable parameters given the data, $p(\\theta \\mid data)$, and not the most probable data given the parameters, $p(data \\mid \\theta) = \\mathcal{L}(\\theta \\mid data)$, since, as we estated above $p(\\theta \\mid data) \\neq p(data \\mid \\theta)$. Does this means traditional frequentist methods are irredeemably flawed? I don't think so. Instead, I think the frequentist approach has been misinterpreted, and everyone can continue using it with no problems as long they are aware of what is that it is estimating, and what are the limitations of said approach. Finally, $p(data)$, in the denominator if equation (7), is considered a normalizing constant which can be calculated by considering the numerator of equation (7) for all the possible values of $\\theta$, since:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "focused-lying",
   "metadata": {},
   "source": [
    "$$p(data)=\\int_{\\theta}^{} p(data \\mid \\theta)p(\\theta) \\; d\\theta \\tag{8}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "arabic-forum",
   "metadata": {},
   "source": [
    "If we observe equations (7) and (8) we can see that we only need to find a way to calculate the *likelihood*, $p(data \\mid \\theta)$, and the *prior*, $p(\\theta)$, to estimate the *posterior* distribution of the parameter given the data, $p(\\theta \\mid data)$, since $p(data)$ depends only on those two as well. We will come back to equation (8) at the end of our estimation process, but for now this proportionality can be expressed as:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eight-semester",
   "metadata": {},
   "source": [
    "$$p(\\theta \\mid data) \\propto p(data \\mid \\theta)p(\\theta)$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "confidential-religious",
   "metadata": {},
   "source": [
    "This, of course, has to be done for every one of our unkown parameters, and the way in which the *likelihood* and the *prior* are defined can varied depending of the characteristics of each parameter we want to estimate. We will try to figure that out in the next section."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "seven-sensitivity",
   "metadata": {},
   "source": [
    "### Getting the *likelihood* and *prior* of our parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "alternate-transformation",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Julia 1.5.4",
   "language": "julia",
   "name": "julia-1.5"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.5.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
