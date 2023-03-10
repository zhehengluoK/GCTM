import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_
import copy
from layer import *


class Scholar(object):

    def __init__(self, config, alpha=1.0, learning_rate=0.001, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, adam_beta1=0.99, adam_beta2=0.999, device=None, seed=None,
                 classify_from_covars=True, model='scholar', topk=1, ablation=None):

        """
        Create the model
        :param config: a dictionary with the model configuration
        :param alpha: hyperparameter for the document representation prior
        :param learning_rate: learning rate for Adam
        :param init_embeddings: a matrix of embeddings to initialize the first layer of the bag-of-words encoder
        :param update_embeddings: if True, update word embeddings during training
        :param init_bg: a vector of empirical log backgound frequencies
        :param update_background: if True, update the background term during training
        :param adam_beta1: first hyperparameter for Adam
        :param adam_beta2: second hyperparameter for Adam
        :param device: (int) the number of the GPU to use
        """

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        # create priors on the hidden state
        self.n_topics = (config["n_topics"])

        if device is None:
            self.device = 'cpu'
        else:
            self.device = 'cuda:' + str(device)

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise, use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        if model == 'scholar':
            self._model = torchScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg, device=self.device, classify_from_covars=classify_from_covars).to(self.device)
        elif model == 'gcnContrastiveScholar':   # only positive or negative sample
            self._model = gcnContrastiveScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg, device=self.device, classify_from_covars=classify_from_covars, topk=topk, ablation=ablation).to(self.device)
        elif model == 'GCNContrastiveScholar':   # full contrastive loss
            self._model = torchGCNContrastiveScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg, device=self.device, classify_from_covars=classify_from_covars, topk=topk).to(self.device)

        # set the criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate, betas=(adam_beta1, adam_beta2))

    def fit(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, eta_bn_prop=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, graphs=None):
        """
        Fit the model to a minibatch of data
        :param X_bow: np.array of document word counts [batch size x vocab size]
        :param X_tfidf: torch.Tensor [batch size x vocab size]
        :param bert_embeddings: embeddings of [batch size x bert_size(1024)]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l1_beta: np.array of prior variances on the topic weights
        :param l1_beta_c: np.array of prior variances on the weights for topic covariates
        :param l1_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X_bow = torch.Tensor(X_bow).to(self.device)
        X_tfidf = X_tfidf.to(self.device)
        bert_embeddings = torch.from_numpy(bert_embeddings).to(self.device)  #
        graphs = [graphs[i].to(self.device) for i in range(len(graphs))]

        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, eta_bn_prop=eta_bn_prop, l1_beta=l1_beta, l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci, graphs=graphs)
        loss, nl, kld, contrast_loss = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), kld.to('cpu').detach().numpy(), contrast_loss.to('cpu').detach().numpy()

    def predict(self, X_tfidf, X_bow, bert_embeddings, PC, TC, eta_bn_prop=0.0, graphs=None):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has been trained on
        batch_size = self.get_batch_size(X_bow)
        Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')

        X_bow = torch.Tensor(X_bow).to(self.device)
        X_tfidf = X_tfidf.to(self.device)
        bert_embeddings = torch.from_numpy(bert_embeddings).to(self.device)  #
        graphs = [graphs[i].to(self.device) for i in range(len(graphs))]

        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _, X_pos, X_neg = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop, graphs=graphs)
        return theta, X_tfidf.to('cpu').detach().numpy(), X_pos.to('cpu').detach().numpy(), X_neg.to('cpu').detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, eta_bn_prop=0.0, n_samples=0, graphs=None):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X_bow)
        if batch_size == 1:
            X_bow = np.expand_dims(X_bow, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X_bow = torch.Tensor(X_bow).to(self.device)
        X_tfidf = X_tfidf.to(self.device)
        bert_embeddings = torch.Tensor(bert_embeddings).to(self.device)
        graphs = [graphs[i].to(self.device) for i in range(len(graphs))]  #

        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)

        if n_samples == 0:
            _, _, _, temp = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop, graphs=graphs)
            loss, NL, KLD, contrast_loss = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, do_average=False, var_scale=1.0, eta_bn_prop=eta_bn_prop, graphs=graphs)
            loss, NL, KLD, contrast_loss = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, do_average=False, var_scale=1.0, eta_bn_prop=eta_bn_prop, graphs=graphs)
                loss, NL, KLD, contrast_loss = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, eta_bn_prop=0.0, graphs=None):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X_bow)
        if batch_size == 1:
            X_bow = np.expand_dims(X_bow, axis=0)
            X_tfidf = np.expand_dims(X_tfidf, axis=0)
            bert_embeddings = np.expand_dims(bert_embeddings, axis=0)   #
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X_bow = torch.Tensor(X_bow).to(self.device)
        X_tfidf = X_tfidf.to(self.device)
        bert_embeddings = torch.Tensor(bert_embeddings).to(self.device)   #
        graphs = [graphs[i].to(self.device) for i in range(len(graphs))]  #

        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)

        theta, _, _, _ = self._model(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop, graphs=graphs)

        return theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_beta(self):
        return self._model.beta

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg
        # return self._model.bg_init.to('cpu').detach().numpy()  #

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to('cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class torchScholar(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu', classify_from_covars=False):
        super(torchScholar, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.classifier_layers = config['classifier_layers']
        self.use_interactions = config['use_interactions']
        self.l1_beta_reg = config['l1_beta_reg']
        self.l1_beta_c_reg = config['l1_beta_c_reg']
        self.l1_beta_ci_reg = config['l1_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.device = device
        self.classify_from_covars = classify_from_covars

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.adapt_bert = nn.Linear(1024, self.vocab_size)  #
        self.embeddings_x_layer = nn.Linear(self.vocab_size * 2, self.words_emb_dim, bias=False)  #
        emb_size = self.words_emb_dim
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            else:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0,
                var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, graphs=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """
        bert_embeddings_, bert_embeddings_pos, bert_embeddings_neg = bert_embeddings
        x_bert = self.adapt_bert(bert_embeddings_)  #
        concate = torch.cat((X_bow, x_bert), dim=1)  #
        en0_x = self.embeddings_x_layer(concate)
        encoder_parts = [en0_x]

        # embed the word counts
        # en0_x = self.embeddings_x_layer(X_bow)
        # encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_drop = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_drop)
        posterior_logvar = self.logvar_layer(encoder_output_drop)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        # posterior_mean_bn = posterior_mean
        # posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X_bow.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_drop = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_drop, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        # eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X_bow, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean(), torch.tensor(0.)
        else:
            return loss, NL, KLD, torch.tensor(0.)

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta.to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon


# This is a simplified version of Contrastive Scholar
# without negative sampling
class gcnContrastiveScholar(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu',
                 classify_from_covars=False, topk=1, ablation=None):
        super(gcnContrastiveScholar, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.classifier_layers = config['classifier_layers']
        self.use_interactions = config['use_interactions']
        self.l1_beta_reg = config['l1_beta_reg']
        self.l1_beta_c_reg = config['l1_beta_c_reg']
        self.l1_beta_ci_reg = config['l1_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.device = device
        self.classify_from_covars = classify_from_covars
        self.ablation = ablation

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.adapt_bert = nn.Linear(1024, self.vocab_size)  #
        self.embeddings_x_layer = nn.Linear(self.vocab_size * 2, self.words_emb_dim, bias=False)  #

        emb_size = self.words_emb_dim  # + 300  #
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        # if bg_init is not None:
        #     self.bg_init = torch.from_numpy(bg_init).to(self.device)  #

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(
                    self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            else:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(
            1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

        self.cos = torch.nn.CosineSimilarity()
        self.topk = topk

        self.gcn = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        self.gcn_pos = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        self.gcn_neg = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        # self.z_pos_bn_layer = nn.BatchNorm1d(self.n_topics, affine=False)
        # self.z_neg_bn_layer = nn.BatchNorm1d(self.n_topics, affine=False)

    def forward(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0,
                var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, graphs=None):
        """
        Do a forward pass of the model

        :param X_bow: np.array of word counts [batch_size x vocab_size]
        :param X_tfidf: shape [batch_size x vocab_size]
        :param bert_embeddings: embedding, [batch_size x bert_size(1024)]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :param graphs: tuple of 4 torch.tensor. e.g. (train_pos_features, train_pos_adj, train_neg_features, train_neg_adj)
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """
        x_bert = self.adapt_bert(bert_embeddings)  #
        concate = torch.cat((X_tfidf, x_bert), dim=1)  #
        en0_x = self.embeddings_x_layer(concate)

        # x_bert = self.adapt_bert(bert_embeddings)  #
        # x_from_w2v = self.embeddings_x_layer(X_tfidf)
        # concate = torch.cat((x_from_w2v, x_bert), dim=1)  #
        # en0_x = concate

        # ????????????????????????????????????????????????????????? bert??????????????? word2vec?????????????????? self.embeddings_x_layer ????????????

        # embed the word count
        # en0_x = self.embeddings_x_layer(X_tfidf)

        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_drop = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_drop)
        posterior_logvar = self.logvar_layer(encoder_output_drop)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        # posterior_mean_bn = posterior_mean
        # posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X_bow.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_drop = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_drop, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # features: V x V, adj: positive word graph
        # features, adj = graphs[0], graphs[1]
        # word-topic matrix, ?????? GCN ??????????????????????????? ????? shape: [vocab size x n_topics]
        # self.beta = self.gcn(features, adj).T
        # eta = torch.matmul(theta, self.beta) + self.bg_init

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        # eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ?????? >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.ablation == 'only_pos':
            # positive part
            features_pos, adj_pos = graphs[0], graphs[1]
            # word-topic matrix, shape[vocab size x n_topics]
            beta_pos = F.softmax(self.gcn_pos(features_pos, adj_pos), dim=0)
            # beta_pos = self.gcn_pos(features_pos, adj_pos)

            # topic distribution   (X_tfidf, shape[batch size x vocab size])
            z_pos = torch.matmul(X_tfidf, beta_pos)
            #
            normalized_z = F.normalize(z)
            normalized_positive_z = F.normalize(z_pos)

            positive_component = self.cos(normalized_z, normalized_positive_z)
            contrastive_loss = -positive_component
        elif self.ablation == 'only_neg':
            # negative part
            features_neg, adj_neg = graphs[2], graphs[3]
            # word-topic matrix, shape[vocab size x n_topics]
            beta_neg = F.softmax(self.gcn_neg(features_neg, adj_neg), dim=0)
            # beta_neg = self.gcn_neg(features_neg, adj_neg)

            # topic distribution   (X: tf-idf, shape[batch size x vocab size])
            z_neg = torch.matmul(X_tfidf, beta_neg)
            # z_neg = torch.matmul(X_tfidf, beta_neg) + torch.matmul(x_bert_neg, beta_neg)

            normalized_z = F.normalize(z)
            normalized_negative_z = F.normalize(z_neg)

            negative_component = self.cos(normalized_z, normalized_negative_z)
            contrastive_loss = negative_component
        else:
            # ???????????????
            contrastive_loss = torch.tensor(0.)

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X_tfidf, Y, X_recon, Y_recon, prior_mean, prior_logvar,
                                                       posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta,
                                                       l1_beta_c, l1_beta_ci, contrastive_loss)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True,
              l1_beta=None, l1_beta_c=None, l1_beta_ci=None, contrastive_loss=None):

        # compute reconstruction loss
        NL = -(X * (X_recon + 1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon + 1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)
        if contrastive_loss is not None: loss += contrastive_loss

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean(), contrastive_loss.mean()
        else:
            return loss, NL, KLD, contrastive_loss

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta.to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon


class torchGCNContrastiveScholar(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu',
                 classify_from_covars=False, topk=1):
        super(torchGCNContrastiveScholar, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.classifier_layers = config['classifier_layers']
        self.use_interactions = config['use_interactions']
        self.l1_beta_reg = config['l1_beta_reg']
        self.l1_beta_c_reg = config['l1_beta_c_reg']
        self.l1_beta_ci_reg = config['l1_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.device = device
        self.classify_from_covars = classify_from_covars

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.adapt_bert = nn.Linear(1024, self.vocab_size)  #
        self.embeddings_x_layer = nn.Linear(self.vocab_size * 2, self.words_emb_dim, bias=False)  #

        emb_size = self.words_emb_dim #+ 300  #
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        # if bg_init is not None:
        #     self.bg_init = torch.from_numpy(bg_init).to(self.device)  #

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(
                    self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            else:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(
            1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

        self.cos = torch.nn.CosineSimilarity()
        self.topk = topk

        self.gcn = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        self.gcn_pos = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        self.gcn_neg = GCN(self.vocab_size, 300, self.n_topics, 0.2)  #
        # self.z_pos_bn_layer = nn.BatchNorm1d(self.n_topics, affine=False)
        # self.z_neg_bn_layer = nn.BatchNorm1d(self.n_topics, affine=False)

    def forward(self, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0,
                var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, graphs=None):
        """
        Do a forward pass of the model

        :param X_bow: np.array of word counts [batch_size x vocab_size]
        :param X_tfidf: shape [batch_size x vocab_size]
        :param bert_embeddings: embeddings of [batch_size x bert_size(1024)]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :param graphs: tuple of 4 torch.tensor. e.g. (train_pos_features, train_pos_adj, train_neg_features, train_neg_adj)
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """
        x_bert = self.adapt_bert(bert_embeddings)  #
        concate = torch.cat((X_tfidf, x_bert), dim=1)  #

        en0_x = self.embeddings_x_layer(concate)

        # x_bert = self.adapt_bert(bert_embeddings)  #
        # x_from_w2v = self.embeddings_x_layer(X_tfidf)
        # concate = torch.cat((x_from_w2v, x_bert), dim=1)  #
        # en0_x = concate

        # ????????????????????????????????????????????????????????? bert??????????????? word2vec?????????????????? self.embeddings_x_layer ????????????

        # embed the word count
        # en0_x = self.embeddings_x_layer(X_tfidf)

        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_drop = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_drop)
        posterior_logvar = self.logvar_layer(encoder_output_drop)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        # posterior_mean_bn = posterior_mean
        # posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X_bow.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_drop = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_drop, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # features: V x V, adj: positive word graph
        # features, adj = graphs[0], graphs[1]
        # word-topic matrix, ?????? GCN ??????????????????????????? ????? shape: [vocab size x n_topics]
        # self.beta = self.gcn(features, adj).T
        # eta = torch.matmul(theta, self.beta) + self.bg_init

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        # eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ?????? >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # graph_pos, graph_neg = graphs  # ??????GCN_pyg

        # positive part
        features_pos, adj_pos = graphs[0], graphs[1]
        # word-topic matrix, shape[vocab size x n_topics]
        beta_pos = F.softmax(self.gcn_pos(features_pos, adj_pos), dim=0)
        # beta_pos = self.gcn_pos(features_pos, adj_pos)
        X_pos = torch.matmul(X_tfidf, adj_pos.to_dense())
        # topic distribution   (X_tfidf, shape[batch size x vocab size])
        z_pos = torch.matmul(X_tfidf, beta_pos)

        normalized_z = F.normalize(z)
        normalized_positive_z = F.normalize(z_pos)

        # negative part
        features_neg, adj_neg = graphs[2], graphs[3]
        # word-topic matrix, shape[vocab size x n_topics]
        beta_neg = F.softmax(self.gcn_neg(features_neg, adj_neg), dim=0)
        # beta_neg = self.gcn_neg(features_neg, adj_neg)
        X_neg = torch.matmul(X_tfidf, adj_neg.to_dense())
        # topic distribution   (X_tfidf, shape[batch size x vocab size])
        z_neg = torch.matmul(X_tfidf, beta_neg)

        normalized_negative_z = F.normalize(z_neg)

        positive_component = self.cos(normalized_z, normalized_positive_z)
        negative_component = self.cos(normalized_z, normalized_negative_z)
        # contrastive_loss = -torch.log(torch.exp(positive_component) /
        #                               (torch.exp(positive_component) + torch.exp(torch.tensor(0.5)) * torch.exp(negative_component)))
        contrastive_loss = -torch.log(torch.exp(positive_component) / (torch.exp(positive_component) + torch.exp(negative_component)))

        # contrastive_loss = torch.tensor(0.)

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X_tfidf, Y, X_recon, Y_recon, prior_mean, prior_logvar,
                                                       posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta,
                                                       l1_beta_c, l1_beta_ci, contrastive_loss), X_pos, X_neg # ??????????????????????????????
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True,
              l1_beta=None, l1_beta_c=None, l1_beta_ci=None, contrastive_loss=None):

        # compute reconstruction loss
        NL = -(X * (X_recon + 1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon + 1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)
        if contrastive_loss is not None: loss += contrastive_loss

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean(), contrastive_loss.mean()
        else:
            return loss, NL, KLD, contrastive_loss

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta.to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon