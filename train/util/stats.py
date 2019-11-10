import torch


def get_top_k(k, decoded, compare, mask):
    """
    Returns the number of elements in ``compare`` for which the corresponding
    entry in ``decoded`` has the value of ``compare`` in its top-k.
    """
    rep_compare = compare.unsqueeze(2).repeat(1, 1, k)
    return torch.sum((decoded.topk(k)[1] == rep_compare).float() * mask.float()).item()


class StatsTracker(object):
    """
    Container for tracking the statistics associated with an epoch. For each of
    a training and validation pass, a new StatsTracker should be instantiated.
    The common use pattern of the class looks as follows::

        for e in range(num_epochs):
            stats = StatsTracker()

            # Add some loss stats
            stats.update_loss(loss)

            # Add accuracy metrics
            stats.update_accuracies(decoded_output, labels, true_labels, mask)

            # Get current average stats
            a, b, c = stats.averages()
    """

    def __init__(self):
        self.loss = 0.

        # Number of correct samples from the view of reconstruction-accuracy.
        self.num_reconstruction_match = 0

        # Number of correct samples from the view of overall-accuracy.
        self.num_overall_match = 0

        self.acc_keys = []
        self.top_k_vals = [1, 2, 5, 10, 20]
        for val in self.top_k_vals:
            self.acc_keys.append("reconstruction_top{}".format(val))
            self.acc_keys.append("overall_top{}".format(val))

        self.acc_map = {}
        for k in self.acc_keys:
            self.acc_map[k] = 0

        # Hold different counters for the number of loss and accuracy attempts.
        # Losses are added in the unit of the average for a minibatch, while
        # accuracy metrics are added for individual samples.
        self.num_loss_attempts = 0
        self.num_match_attempts = 0

        self.running_top1 = 0
        self.running_top5 = 0

    def averages(self):
        """
        Returns average loss, reconstruction-accuracy, and overall-accuracy
        since this ``StatsTracker`` was instantiated.
        """
        avg_loss = self.loss / self.num_loss_attempts

        for k in self.acc_map:
            self.acc_map[k] /= self.num_match_attempts

        return avg_loss, self.acc_map

    def running_averages(self):
        """
        Returns running average loss, top-1 overall accuracy, and top-5
        overall accuracy.
        """
        avg_loss = self.loss / self.num_loss_attempts
        top1 = self.running_top1 / self.num_match_attempts
        top5 = self.running_top5 / self.num_match_attempts
        return avg_loss, top1, top5

    def update_accuracies(self, decoded, base_model_outputs, true_labels, mask):
        """
        Calculates the number of decoded outputs that match (1) the outputs
        from the base model and (2) the true labels associated with the decoded
        sample. These results are maintained for later aggregate statistics.
        """
        self.num_match_attempts += decoded.size(0)
        max_decoded = torch.max(decoded, dim=2)[1]
        max_outputs = torch.max(base_model_outputs, dim=2)[1]

        self.acc_map["reconstruction_top1"] += torch.sum(
            (max_decoded == max_outputs).float() * mask.float()).item()
        top1_correct = torch.sum((max_decoded == true_labels).float() * mask.float()).item()
        self.acc_map["overall_top1"] += top1_correct
        self.running_top1 += top1_correct

        # Ignore first of top_k_vals because we already covered it above.
        for k in self.top_k_vals[1:]:
            if decoded.size(-1) < k:
                break

            acc_mask = mask.unsqueeze(2).repeat(1, 1, k)
            self.acc_map["reconstruction_top{}".format(
                k)] += get_top_k(k, decoded, max_outputs, acc_mask)
            overall_correct = get_top_k(k, decoded, true_labels, acc_mask)
            self.acc_map["overall_top{}".format(k)] += overall_correct
            if k == 5:
                self.running_top5 += overall_correct

    def update_loss(self, loss):
        """
        Adds ``loss`` to the current aggregate loss for this epoch.
        """
        self.loss += loss
        self.num_loss_attempts += 1
