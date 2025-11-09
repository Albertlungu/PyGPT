import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.transformer.output_layer import OutputLayer
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_block import TransformerBlock

class CrossEntropyLoss:
    """
    Function that tells the model how wrong its predictions are. 
    Cross-entropy measures the difference between the model's prediction (from output_layer) and the true next token.
    """
    def __init__(self, ignore_index = None, reduction = 'mean'):
        """
        Initializing CrossEntropyLoss instance attributes

        Args:
            ignore_index (int, optional): Which index to ignore. Defaults to None.
            reduction (str, optional): Type of reduction, 'sum' or 'mean'. Defaults to 'mean'.
        """
        self.ignore_index = ignore_index 
        self.reduction = reduction

    def fwd(self, logits, targets):
        """
        Forward pass for the cross entropy loss function. Calculates the actual loss for each token.

        Args:
            logits (np.ndarray): Output of the OutputLayer class.
            targets (np.ndarray): The true "next-token" index that the model is supposed to predict. Shape: (batch, seq_len)

        Returns:
            np.float64: Loss of my model I guess

        Raises:
            TypeError
        """
        targets = np.array(targets)

        self.batch_size, self.seq_len, self.vocab_size = logits.shape

        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            logits = logits[mask]
            targets = targets[mask]

        # Using the cross-entropy mathematical definition to find the loss
        max_logits = np.max(logits, axis = -1, keepdims= True)
        shifted_logits = logits - max_logits

        exp_logits = np.exp(shifted_logits)
        sum_exp = np.sum(exp_logits, axis = -1, keepdims=True)
        log_sum_exp = np.log(sum_exp)

        log_probs = shifted_logits - log_sum_exp

        selected_log_probs = log_probs[np.arange(len(targets)), targets]

        neg_log_prob = -selected_log_probs
        

        valid_mask = (targets != self.ignore_index)
        masked_loss = neg_log_prob * valid_mask
        num_valid = np.sum(valid_mask)

        if self.reduction == 'mean':
            loss = np.sum(masked_loss)/num_valid
        elif self.reduction == 'sum':
            loss = np.sum(masked_loss)
        else: 
            raise TypeError("Please enter either 'mean' or 'sum' as a reduction")
        
        return loss

    def backward(self, logits, targets, probs):
        """
        The backward pass for cross-entropy loss function. Returns how much the loss would change if you changed the raw output scores (logits) of the model, for each token and each possible vocab item.

        Args:
            logits (np.ndarray): Logits from the OutputLayer class
            targets (np.ndarray): The true "next-token" index that the model is supposed to predict. Shape: (batch, seq_len)
            probs (np.ndarray): The probabilities calculated from the softmax of the logits from the OutputLayer class

        Returns:
            np.ndarray of shape (batch_size, seq_len, vocab_size), dtype np.float64: 
                The gradient of the cross-entropy loss with respect to the logits. Each element represents how much the loss would increase or decrease if the corresponding logit were increased by a small amount. This gradient is used to propagate errors backward through the network during training.
        """
        targets = np.array(targets)

        batch_size, seq_len, vocab_size = logits.shape

        one_hot_targets = np.zeros_like(logits)
        one_hot_flat = one_hot_targets.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        one_hot_flat[np.arange(len(targets)), targets] = 1
        one_hot_targets = one_hot_flat.reshape(batch_size, seq_len, vocab_size)

        d_logits = probs - one_hot_targets

        d_logits = d_logits / (batch_size * seq_len)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            d_logits = d_logits * mask[:, :, None]  # broadcast mask over vocab dimension

        return d_logits