import os
import argparse
import json
import matplotlib.pyplot as plt

##############################################################################
# Plot Training Losses
##############################################################################
def plot_train_loss(fname):
  x = []
  loss = []

  with open(fname, 'r') as fp:
    for l in fp.readlines():
      if l.startswith("{'loss':"):        # loss log down
        l = l.strip().replace("\'", "\"")
        data = json.loads(l)
        x.append(data['epoch'])
        loss.append(data['loss'])

  return x, loss


##############################################################################
# Plot Testing Losses and macro-F1
##############################################################################
def plot_testloss_f1(pfeiffer, houlsby, parallel, ylabel, title, figloc, figtitle):
  fig, ax = plt.subplots()
  ax.plot(x, pfeiffer, '#425066', label='Pfeiffer')
  ax.plot(x, houlsby, '#12b5cb', label='Houlsby')
  ax.plot(x, parallel, '#e52592', label='Parallel')
  ax.set_xlabel('Epoch')
  ax.set_ylabel(ylabel)
  ax.legend(loc=figloc)
  ax.set_title(title)
  # plt.show()
  plt.savefig(figtitle)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir',    default='log',    help='Log directory')
  parser.add_argument('--task',       default='aws',    help='task name')

  args = parser.parse_args()

  pfeiffer_fn = os.path.join(args.log_dir, 'log_' + args.task + '_pfeiffer.txt')
  houlsby_fn = os.path.join(args.log_dir,  'log_' + args.task + '_houlsby.txt')
  parallel_fn = os.path.join(args.log_dir, 'log_' + args.task + '_parallel.txt')

  x1, l1 = plot_train_loss(pfeiffer_fn)
  x2, l2 = plot_train_loss(houlsby_fn)
  x3, l3 = plot_train_loss(parallel_fn)

  ##############################################################################
  # Plot Training Losses

  fig, ax = plt.subplots()
  ax.plot(x1, l1, '#425066', label='Pfeiffer')
  ax.plot(x2, l2, '#12b5cb', label='Houlsby')
  ax.plot(x3, l3, '#e52592', label='Parallel')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Training Loss')
  ax.legend(loc='upper right')
  #ax.set_title('Training Loss for Task %s' % args.task)
  #plt.show()
  plt.savefig(args.task + "_train_loss.png", dpi=400)



  ##############################################################################
  # Plot Testing Losses and macro-F1

  ####### IMDb #######
  # Macro F-1
  x = np.arange(0, 10, 1)
  pfeiffer = np.array([94.45, 94.58, 94.99, 95.42, 95.28, 95.31, 95.52, 95.57, 95.45, 95.56])
  houlsby = np.array([94.75, 95.36, 95.13, 95.38, 95.34, 95.52, 95.45, 95.42, 95.54, 95.56])
  parallel = np.array([94.76, 95.13, 95.17, 95.18, 95.01, 95.05, 94.98, 94.93, 94.99, 94.94])
  pfeiffer /= 100
  houlsby /= 100
  parallel /= 100
  plot_testloss_f1(pfeiffer, houlsby, parallel, 'Testing Macro-F1', '', 'lower right', 'IMDB-F1')

  # Test Loss
  pfeiffer = np.array([0.1533, 0.1553, 0.1370, 0.1307, 0.1328, 0.1419, 0.1395, 0.1486, 0.1533, 0.1518])
  houlsby = np.array([0.1454, 0.1349, 0.1318, 0.1349, 0.1275, 0.1367, 0.1358, 0.1478, 0.1475, 0.1517])
  parallel = np.array([0.1486, 0.1539, 0.1409, 0.1603, 0.1590, 0.2421, 0.2404, 0.2956, 0.3187, 0.3514])
  plot_testloss_f1(pfeiffer, houlsby, parallel, 'Testing Loss', '', 'upper left', 'IMDB-TestLoss')

  ####### Helpfulness #######
  # Macro F-1
  pfeiffer = np.array([59.38, 64.23, 65.03, 69.18, 70.68, 65.88, 67.22, 70.16, 70.22, 69.21])
  houlsby = np.array([57.83, 66.85, 69.87, 64.88, 70.17, 64.84, 67.79, 69.92, 70.40, 69.93])
  parallel = np.array([54.73, 65.37, 66.12, 62.65, 67.60, 66.61, 65.12, 68.24, 67.61, 67.40])
  pfeiffer /= 100
  houlsby /= 100
  parallel /= 100
  plot_testloss_f1(pfeiffer, houlsby, parallel, 'Testing Macro-F1', '', 'lower right', 'Helpfulness-F1')

  # Test Loss
  pfeiffer = np.array([0.3181, 0.3228, 0.3004, 0.3013, 0.3010, 0.3052, 0.3110, 0.3037, 0.3100, 0.3089])
  houlsby = np.array([0.3230, 0.3017, 0.2997, 0.3096, 0.3020, 0.3135, 0.3001, 0.3023, 0.3113, 0.3126])
  parallel = np.array([0.3291, 0.3140, 0.3152, 0.3492, 0.3305, 0.3350, 0.3661, 0.3779, 0.4387, 0.4547])
  plot_testloss_f1(pfeiffer, houlsby, parallel, 'Testing Loss', '', 'upper left', 'Helpfulness-TestLoss')






