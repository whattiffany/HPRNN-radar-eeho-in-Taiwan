
import matplotlib.pyplot as plt
if __name__ == '__main__':

    train_loss = []
    test_loss = []
    with open("loss.txt", 'r') as file_1:
        lines = file_1.readlines()
        for i,line in enumerate(lines, start=1):
            if ("model train loss") in line:
                print("model train loss")
                train =line.split("loss",1)[-1]
                train_loss.append(float(train))
            elif ("model test loss") in line:
                print("model test loss")
                test =line.split("loss",1)[-1]
                test_loss.append(float(test))
    title = "Loss(batch_size 1)"
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='train_loss')

    plt.xlabel("epoch")
    plt.legend(['train', 'test'], loc='upper right')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()