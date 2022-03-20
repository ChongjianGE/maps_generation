import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from tensorboardX import SummaryWriter
from util import util
import datetime
import ipdb

if __name__ == '__main__':
    opt = TrainOptions().parse()
    #ipdb.set_trace()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    total_iters = 0

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    writer = SummaryWriter('./train_output/runs/' + opt.name)
    step_per_batch = dataset_size / opt.batch_size
    load_flag = False
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            vis = model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                writer.add_image('vis', util.tensor2im(vis).transpose((2,0,1)), total_iters)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                t_per_batch = (time.time() - iter_start_time)
                batch_left = (step_per_batch-total_iters % step_per_batch) + step_per_batch*(opt.n_epochs + opt.n_epochs_decay + 1 - epoch)
                eta = t_per_batch * batch_left
                eta = str(datetime.timedelta(seconds=int(eta)))
                util.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, eta, log_name)
                writer.add_scalar('G_GAN', losses['G_GAN'], total_iters)
                writer.add_scalar('G_L1', losses['G_L1'], total_iters)
                writer.add_scalar('D_real', losses['D_real'], total_iters)
                writer.add_scalar('D_fake', losses['D_fake'], total_iters)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            if epoch > 1:
                load_flag = False

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
