import torch
import plot
import model

def test_data_process():
    test_loader = plot.test_Loader
    return test_loader


def test_model_process(model, test_data):
    test_acc=0.0
    test_num=0
    model.eval()
    with torch.no_grad():
        for test_x,test_y in test_data:
            output=model(test_x)
            pre_label=torch.argmax(output,dim=1)
            test_acc+=torch.sum(pre_label==test_y)
            test_num+=test_x.size(0)

            label=test_y.item()
            result=pre_label.item()
            if label==result:
                print("预测值：",result,"-------","真实值",label)
            else:
                print("预测值：",result,"-----------------------","真实值",label)
    test_avd_acc=test_acc/test_num
    print("测试准确率:",test_avd_acc)


if __name__ == '__main__':
    test_data=test_data_process()
    my_model=model.LeNet()
    my_model.load_state_dict(torch.load('best_model.pth'))
    test_model_process(my_model,test_data)