import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import model
import utility

if __name__ == '__main__':
    BATCH_SIZE = 1
    INITIAL_LEARNING_RATE = 0.0001
    EPOCHS = 500
    adobe_train_path = "/Users/posoo/GitRepos/adobeset/train"

    '''
    define our device as the first visible cuda device if we have CUDA available
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    """
    using pre-trained VGG-16 to output conv_4_3 feature
    """
    vgg16 = models.vgg16(pretrained=True)
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:23])

    """
    prepare training set
    """
    transform_to_Tensor = transforms.Compose([transforms.ToTensor()])
    train_dataset = utility.AdobeDataset(adobe_train_path, transform=transform_to_Tensor)
    print("DATASET SIZE: {}".format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    """
    initialize net models and optimizer
    """
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    flow_computation = model.UNet(6, 4)
    flow_computation.to(device)
    arbitrary_time_flow_interpolation = model.UNet(20, 5)
    arbitrary_time_flow_interpolation.to(device)

    backward_warping = model.BackwardWarping(352, 352, device)

    learning_rate = INITIAL_LEARNING_RATE
    optimizer = optim.Adam(list(flow_computation.parameters()) +
                           list(arbitrary_time_flow_interpolation.parameters()),
                           lr=learning_rate)

    """
    start training
    """
    for epoch in range(EPOCHS):
        for i, (I_0, I_t, I_1, t) in enumerate(train_loader, 0):
            I_0 = I_0.to(device)
            I_t = I_t.to(device)
            I_1 = I_1.to(device)

            """
            zero the parameter gradients
            """
            optimizer.zero_grad()

            """
            calculate loss
            """
            flow_outputs = flow_computation(torch.cat((I_0, I_1), dim=1))
            F_0_1 = flow_outputs[:, :2, :, :]
            F_1_0 = flow_outputs[:, 2:, :, :]

            slomo_cp = model.SuperSlomo(arbitrary_time_flow_interpolation, backward_warping, I_0, I_1, device, F_0_1,
                                        F_1_0, True)
            g_t_0_hat, g_t_1_hat, I_t_hat, g_0_hat, g_1_hat = slomo_cp.forward(2, t)
            loss_reconstruction = l1_loss(I_t_hat, I_t)
            loss_perceptual = l2_loss(vgg16_conv_4_3(I_t_hat), vgg16_conv_4_3(I_t_hat))
            loss_warping = l1_loss(I_1, g_0_hat) + l1_loss(I_0, g_1_hat) + \
                           l1_loss(I_t, g_t_0_hat) + l1_loss(I_t, g_t_1_hat)

            loss_smooth = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + \
                          torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :])) + \
                          torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_total = 0.8 * loss_reconstruction + 0.005 * loss_perceptual + 0.4 * loss_warping + 1 * loss_smooth

            """
            BP update
            """
            loss_total.backward()
            optimizer.step()

            """
            print every 200 iterations
            """
            if i % 200 == 0:
                print("EPOCHS: {}/{} - {}, lr: {}, lp: {}, lw: {}, ls: {}, ltotal:{}".format(epoch, EPOCHS, i,
                                                                                             loss_reconstruction,
                                                                                             loss_perceptual,
                                                                                             loss_warping,
                                                                                             loss_smooth,
                                                                                             loss_total))

            """
            Learning Rate decreased by a factor of 10 every 200 epochs
            """
            if epoch % 200 == 199:
                learning_rate = learning_rate / 10
                optimizer = optim.Adam(list(flow_computation.parameters()) +
                                       list(arbitrary_time_flow_interpolation.parameters()),
                                       lr=learning_rate)

            """
            Save checkpoint per 50 epochs
            """
            if epoch % 50 == 49:
                checkpoint_dict = {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "loss_total": loss_total,
                    "loss_reconstruction": loss_reconstruction,
                    "loss_perceptual": loss_perceptual,
                    "loss_warping": loss_warping,
                    "loss_smooth": loss_smooth,
                    "flow_computation_state": flow_computation.state_dict(),
                    "arbitrary_time_interpolation_state": arbitrary_time_flow_interpolation.state_dict()
                }
                torch.save(checkpoint_dict, "superslomo-{}.ckpt".format(int(time.time())))
