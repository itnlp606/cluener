from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.08, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cycle_schedule(optimizer, cycle_steps, last_epoch=-1):
    def lr_lambda(current_step):
        return max(0.05, 1 - (current_step % cycle_steps + 1)/cycle_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch)