
import numpy as np
import matplotlib.pyplot as plt

train_epoch = 20001
valid_epoch = 20000
EPOCH = 20000

def main():

#****************************************************************************
# 1.1.2
#****************************************************************************
    # r1 = np.load("data_output/1.1.2.1.npz")
    # train_loss_x1 = np.arange(train_epoch) + 1
    # train_loss_y1 = r1["train_loss"]
    # valid_loss_x1 = np.arange(valid_epoch) + 1
    # valid_loss_y1 = r1["valid_loss"]
    # test_loss_x1 = np.arange(EPOCH) + 1
    # test_loss_y1 = r1["test_loss"]
    # train_acc_x1 = np.arange(train_epoch) + 1
    # train_acc_y1 = r1["train_acc"]
    # valid_acc_x1 = np.arange(valid_epoch) + 1
    # valid_acc_y1 = r1["valid_acc"]
    # test_acc_x1 = np.arange(EPOCH) + 1
    # test_acc_y1 = r1["test_acc"]

    # r2 = np.load("data_output/1.1.2.2.npz")
    # train_loss_x2 = np.arange(train_epoch) + 1
    # train_loss_y2 = r2["train_loss"]
    # valid_loss_x2 = np.arange(valid_epoch) + 1
    # valid_loss_y2 = r2["valid_loss"]
    # test_loss_x2 = np.arange(EPOCH) + 1
    # test_loss_y2 = r2["test_loss"]
    # train_acc_x2 = np.arange(train_epoch) + 1
    # train_acc_y2 = r2["train_acc"]
    # valid_acc_x2 = np.arange(valid_epoch) + 1
    # valid_acc_y2 = r2["valid_acc"]
    # test_acc_x2 = np.arange(EPOCH) + 1
    # test_acc_y2 = r2["test_acc"]

    r3 = np.load("npz_prev_2/1.1.2.3.npz")
    train_loss_x3 = np.arange(train_epoch) + 1
    train_loss_y3 = r3["train_loss"]
    valid_loss_x3 = np.arange(valid_epoch) + 1
    valid_loss_y3 = r3["valid_loss"]
    test_loss_x3 = np.arange(EPOCH) + 1
    test_loss_y3 = r3["test_loss"]
    train_acc_x3 = np.arange(train_epoch) + 1
    train_acc_y3 = r3["train_acc"]
    valid_acc_x3 = np.arange(valid_epoch) + 1
    valid_acc_y3 = r3["valid_acc"]
    test_acc_x3 = np.arange(EPOCH) + 1
    test_acc_y3 = r3["test_acc"]

    plt.figure(1)
    plt.clf()
    # plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, lr = 0.001")
    # plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, lr = 0.003")
    # plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, lr = 0.01")
    # plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, lr = 0.001")
    # plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, lr = 0.003")
    plt.plot(valid_loss_x3, valid_loss_y3, label = "valid, learning rate = 0.01")
    plt.plot([5765,5765],[0.25,valid_loss_y3[5765]], '--y')
    plt.plot(5765,valid_loss_y3[5765], 'yo', label = "early stopping point")
    plt.legend()

    plt.title("Validation Loss versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("loss_early_stopping.png")
    # plt.show()

    plt.figure(2)
    plt.clf()
    # plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, lr = 0.001")
    # plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, lr = 0.003")
    # plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, lr = 0.01")
    # plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, lr = 0.001")
    # plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, lr = 0.003")
    plt.plot(valid_acc_x3, valid_acc_y3, label = "valid, learning rate = 0.01")
    plt.plot([1620,1620],[0.85,valid_acc_y3[1620]], '--y')
    plt.plot(1620,valid_acc_y3[1620], 'yo', label = "early stopping point")
    plt.legend()

    plt.title("Validation Accuracy versus Number of Epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.savefig("acc_early_stopping.png")
    # plt.show()

    # plt.figure(3)
    # plt.clf()
    # plt.semilogx(test_loss_x1, test_loss_y1, label = "test, lr = 0.001")
    # plt.semilogx(test_loss_x2, test_loss_y2, label = "test, lr = 0.003")
    # plt.semilogx(test_loss_x3, test_loss_y3, label = "test, lr = 0.01")
    # plt.legend()

    # plt.title("Test Loss versus Number of Epochs")
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Loss")
    # plt.savefig("picture/1.1.2/1.1.2_te_loss.png")

    # plt.figure(4)
    # plt.clf()
    # plt.semilogx(test_acc_x1, test_acc_y1, label = "test, lr = 0.001")
    # plt.semilogx(test_acc_x2, test_acc_y2, label = "test, lr = 0.003")
    # plt.semilogx(test_acc_x3, test_acc_y3, label = "test, lr = 0.01")
    # plt.legend()

    # plt.title("Test Accuracy versus Number of Epochs")
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Accuracy")
    # plt.savefig("picture/1.1.2/1.1.2_te_acc.png")


# #****************************************************************************
# # 1.2.1
# #****************************************************************************
#     r1 = np.load("data_output/1.1.2.3.npz")
#     train_loss_x1 = np.arange(train_epoch) + 1
#     train_loss_y1 = r1["train_loss"]
#     valid_loss_x1 = np.arange(valid_epoch) + 1
#     valid_loss_y1 = r1["valid_loss"]
#     test_loss_x1 = np.arange(EPOCH) + 1
#     test_loss_y1 = r1["test_loss"]
#     train_acc_x1 = np.arange(train_epoch) + 1
#     train_acc_y1 = r1["train_acc"]
#     valid_acc_x1 = np.arange(valid_epoch) + 1
#     valid_acc_y1 = r1["valid_acc"]
#     test_acc_x1 = np.arange(EPOCH) + 1
#     test_acc_y1 = r1["test_acc"]

#     r2 = np.load("data_output/1.2.1.2.npz")
#     train_loss_x2 = np.arange(train_epoch) + 1
#     train_loss_y2 = r2["train_loss"]
#     valid_loss_x2 = np.arange(valid_epoch) + 1
#     valid_loss_y2 = r2["valid_loss"]
#     test_loss_x2 = np.arange(EPOCH) + 1
#     test_loss_y2 = r2["test_loss"]
#     train_acc_x2 = np.arange(train_epoch) + 1
#     train_acc_y2 = r2["train_acc"]
#     valid_acc_x2 = np.arange(valid_epoch) + 1
#     valid_acc_y2 = r2["valid_acc"]
#     test_acc_x2 = np.arange(EPOCH) + 1
#     test_acc_y2 = r2["test_acc"]

#     r3 = np.load("data_output/1.2.1.3.npz")
#     train_loss_x3 = np.arange(train_epoch) + 1
#     train_loss_y3 = r3["train_loss"]
#     valid_loss_x3 = np.arange(valid_epoch) + 1
#     valid_loss_y3 = r3["valid_loss"]
#     test_loss_x3 = np.arange(EPOCH) + 1
#     test_loss_y3 = r3["test_loss"]
#     train_acc_x3 = np.arange(train_epoch) + 1
#     train_acc_y3 = r3["train_acc"]
#     valid_acc_x3 = np.arange(valid_epoch) + 1
#     valid_acc_y3 = r3["valid_acc"]
#     test_acc_x3 = np.arange(EPOCH) + 1
#     test_acc_y3 = r3["test_acc"]

#     plt.figure(1)
#     plt.clf()
#     plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_loss_x3, valid_loss_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_tv_loss.png")

#     plt.figure(2)
#     plt.clf()
#     plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_acc_x3, valid_acc_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_tv_acc.png")

#     plt.figure(3)
#     plt.clf()
#     plt.semilogx(test_loss_x1, test_loss_y1, label = "test, hs = 1000")
#     plt.semilogx(test_loss_x2, test_loss_y2, label = "test, hs = 500")
#     plt.semilogx(test_loss_x3, test_loss_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_te_loss.png")

#     plt.figure(4)
#     plt.clf()
#     plt.semilogx(test_acc_x1, test_acc_y1, label = "test, hs = 1000")
#     plt.semilogx(test_acc_x2, test_acc_y2, label = "test, hs = 500")
#     plt.semilogx(test_acc_x3, test_acc_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_te_acc.png")


# #****************************************************************************
# # 1.2.2
# #****************************************************************************
#     r1 = np.load("data_output/1.1.2.3.npz")
#     train_loss_x1 = np.arange(train_epoch) + 1
#     train_loss_y1 = r1["train_loss"]
#     valid_loss_x1 = np.arange(valid_epoch) + 1
#     valid_loss_y1 = r1["valid_loss"]
#     test_loss_x1 = np.arange(EPOCH) + 1
#     test_loss_y1 = r1["test_loss"]
#     train_acc_x1 = np.arange(train_epoch) + 1
#     train_acc_y1 = r1["train_acc"]
#     valid_acc_x1 = np.arange(valid_epoch) + 1
#     valid_acc_y1 = r1["valid_acc"]
#     test_acc_x1 = np.arange(EPOCH) + 1
#     test_acc_y1 = r1["test_acc"]

#     r2 = np.load("data_output/1.2.1.2.npz")
#     train_loss_x2 = np.arange(train_epoch) + 1
#     train_loss_y2 = r2["train_loss"]
#     valid_loss_x2 = np.arange(valid_epoch) + 1
#     valid_loss_y2 = r2["valid_loss"]
#     test_loss_x2 = np.arange(EPOCH) + 1
#     test_loss_y2 = r2["test_loss"]
#     train_acc_x2 = np.arange(train_epoch) + 1
#     train_acc_y2 = r2["train_acc"]
#     valid_acc_x2 = np.arange(valid_epoch) + 1
#     valid_acc_y2 = r2["valid_acc"]
#     test_acc_x2 = np.arange(EPOCH) + 1
#     test_acc_y2 = r2["test_acc"]

#     r3 = np.load("data_output/1.2.1.3.npz")
#     train_loss_x3 = np.arange(train_epoch) + 1
#     train_loss_y3 = r3["train_loss"]
#     valid_loss_x3 = np.arange(valid_epoch) + 1
#     valid_loss_y3 = r3["valid_loss"]
#     test_loss_x3 = np.arange(EPOCH) + 1
#     test_loss_y3 = r3["test_loss"]
#     train_acc_x3 = np.arange(train_epoch) + 1
#     train_acc_y3 = r3["train_acc"]
#     valid_acc_x3 = np.arange(valid_epoch) + 1
#     valid_acc_y3 = r3["valid_acc"]
#     test_acc_x3 = np.arange(EPOCH) + 1
#     test_acc_y3 = r3["test_acc"]

#     plt.figure(1)
#     plt.clf()
#     plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_loss_x3, valid_loss_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_tv_loss.png")

#     plt.figure(2)
#     plt.clf()
#     plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_acc_x3, valid_acc_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_tv_acc.png")

#     plt.figure(3)
#     plt.clf()
#     plt.semilogx(test_loss_x1, test_loss_y1, label = "test, hs = 1000")
#     plt.semilogx(test_loss_x2, test_loss_y2, label = "test, hs = 500")
#     plt.semilogx(test_loss_x3, test_loss_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_te_loss.png")

#     plt.figure(4)
#     plt.clf()
#     plt.semilogx(test_acc_x1, test_acc_y1, label = "test, hs = 1000")
#     plt.semilogx(test_acc_x2, test_acc_y2, label = "test, hs = 500")
#     plt.semilogx(test_acc_x3, test_acc_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_te_acc.png")


# #****************************************************************************
# # 1.3.1
# #****************************************************************************
#     r1 = np.load("data_output/1.1.2.3.npz")
#     train_loss_x1 = np.arange(train_epoch) + 1
#     train_loss_y1 = r1["train_loss"]
#     valid_loss_x1 = np.arange(valid_epoch) + 1
#     valid_loss_y1 = r1["valid_loss"]
#     test_loss_x1 = np.arange(EPOCH) + 1
#     test_loss_y1 = r1["test_loss"]
#     train_acc_x1 = np.arange(train_epoch) + 1
#     train_acc_y1 = r1["train_acc"]
#     valid_acc_x1 = np.arange(valid_epoch) + 1
#     valid_acc_y1 = r1["valid_acc"]
#     test_acc_x1 = np.arange(EPOCH) + 1
#     test_acc_y1 = r1["test_acc"]

#     r2 = np.load("data_output/1.2.1.2.npz")
#     train_loss_x2 = np.arange(train_epoch) + 1
#     train_loss_y2 = r2["train_loss"]
#     valid_loss_x2 = np.arange(valid_epoch) + 1
#     valid_loss_y2 = r2["valid_loss"]
#     test_loss_x2 = np.arange(EPOCH) + 1
#     test_loss_y2 = r2["test_loss"]
#     train_acc_x2 = np.arange(train_epoch) + 1
#     train_acc_y2 = r2["train_acc"]
#     valid_acc_x2 = np.arange(valid_epoch) + 1
#     valid_acc_y2 = r2["valid_acc"]
#     test_acc_x2 = np.arange(EPOCH) + 1
#     test_acc_y2 = r2["test_acc"]

#     r3 = np.load("data_output/1.2.1.3.npz")
#     train_loss_x3 = np.arange(train_epoch) + 1
#     train_loss_y3 = r3["train_loss"]
#     valid_loss_x3 = np.arange(valid_epoch) + 1
#     valid_loss_y3 = r3["valid_loss"]
#     test_loss_x3 = np.arange(EPOCH) + 1
#     test_loss_y3 = r3["test_loss"]
#     train_acc_x3 = np.arange(train_epoch) + 1
#     train_acc_y3 = r3["train_acc"]
#     valid_acc_x3 = np.arange(valid_epoch) + 1
#     valid_acc_y3 = r3["valid_acc"]
#     test_acc_x3 = np.arange(EPOCH) + 1
#     test_acc_y3 = r3["test_acc"]

#     plt.figure(1)
#     plt.clf()
#     plt.semilogx(train_loss_x1, train_loss_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_loss_x2, train_loss_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_loss_x3, train_loss_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_loss_x1, valid_loss_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_loss_x2, valid_loss_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_loss_x3, valid_loss_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_tv_loss.png")

#     plt.figure(2)
#     plt.clf()
#     plt.semilogx(train_acc_x1, train_acc_y1, '--', label = "train, hs = 1000")
#     plt.semilogx(train_acc_x2, train_acc_y2, '--', label = "train, hs = 500")
#     plt.semilogx(train_acc_x3, train_acc_y3, '--', label = "train, hs = 100")
#     plt.semilogx(valid_acc_x1, valid_acc_y1, label = "valid, hs = 1000")
#     plt.semilogx(valid_acc_x2, valid_acc_y2, label = "valid, hs = 500")
#     plt.semilogx(valid_acc_x3, valid_acc_y3, label = "valid, hs = 100")
#     plt.legend()

#     plt.title("Training and Validation Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_tv_acc.png")

#     plt.figure(3)
#     plt.clf()
#     plt.semilogx(test_loss_x1, test_loss_y1, label = "test, hs = 1000")
#     plt.semilogx(test_loss_x2, test_loss_y2, label = "test, hs = 500")
#     plt.semilogx(test_loss_x3, test_loss_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Loss versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Loss")
#     plt.savefig("picture/1.2.1/1.2.1_te_loss.png")

#     plt.figure(4)
#     plt.clf()
#     plt.semilogx(test_acc_x1, test_acc_y1, label = "test, hs = 1000")
#     plt.semilogx(test_acc_x2, test_acc_y2, label = "test, hs = 500")
#     plt.semilogx(test_acc_x3, test_acc_y3, label = "test, hs = 100")
#     plt.legend()

#     plt.title("Test Accuracy versus Number of Epochs")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig("picture/1.2.1/1.2.1_te_acc.png")


if __name__ == '__main__':
	main()
