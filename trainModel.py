def train_model(model, dataloaders,criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_loss_history = []
    best_loss= np.inf
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(0,num_epochs):
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0             

            for i ,[inputs, labels] in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        action_loss=loss
                    preds = outputs
                   
                    if phase == 'train':
                        action_loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            writer.add_scalar('Loss/{}'.format(phase),epoch_loss,epoch)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, val_loss_history
