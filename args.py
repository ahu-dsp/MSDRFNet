class args():
    # training args
    epochs=4
    batch_size =128
    HEIGHT =64
    WIDTH =64
    downsample = ['stride', "avgpool", "maxpool"]
    dataset_ir = "/Share/home/Z21301084/test/RFN1/results/M-SPE/SPECT"
    dataset_vi= "/Share/home/Z21301084/test/RFN1/results/M-SPE/MRI"

    dataset= "/Share/home/Z21301084/test/RFN1/MMI/COCO-train2017"
    save_model_dir_encoder = "models/model"
    # save_loss_dir = "models/loss"

    cuda = 1
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
    grad_weight = [1, 10, 100, 1000, 10000]

    lr =10e-4  # "learning rate, default is 0.001"
    # lr_light = 10e-4  # "learning rate, default is 0.001"
    log_interval = 10  # "number of images after which the training loss is logged, default is 500"
    resume = None

    # for test, model_default is the model used in paper
    model_default = './model/UNFusion.pth'
    model_deepsuper = 'UNFusion.model'

    save_fusion_model_onestage="./onestage/model"
    save_loss_dir_onestage="./onestage/loss"
    resume_fusion_model=None
    save_fusion_model = "/Share/home/Z21301084/test/RFN1/UN1/model/function/"
    save_loss_dir = '/Share/home/Z21301084/test/RFN1/UN1/models/loss/function/'
    train_num=1000