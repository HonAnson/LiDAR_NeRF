def forward(self, o, d):
    emb_x = self.positional_encoding(o, self.embedding_dim_pos)
    emb_d = self.positional_encoding(d, self.embedding_dim_dir)
    input = torch.hstack((emb_x,emb_d)).to(dtype=torch.float32)
    temp = self.block1(input)
    input2 = torch.hstack((temp, input)).to(dtype=torch.float32) # add skip input
    output = self.block2(input2)
    return output, o  # Return both output and input position




def train(model, optimizer, scheduler, dataloader, device='cuda', num_epoch=int(1e5), num_bins=100, eikonal_weight=0.1):
    training_losses = []
    loss_MSE = nn.MSELoss()
    num_batch_in_data = len(dataloader)
    count = 0
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):
            # parse the batch
            centers = batch[:,0:3]
            directions = batch[:,3:6]
            depth = batch[:,6]

            sample_pos, sample_dir, depth_target = getSamplesAndTarget(centers, directions, depth, num_bins=num_bins)
            sample_pos = sample_pos.to(device)
            sample_dir = sample_dir.to(device)
            depth_target = depth_target.to(device, dtype=torch.float32)
            
            # Enable gradient computation for input positions
            sample_pos.requires_grad_(True)
            
            # inference
            xyz_pred, input_pos = model(sample_pos, sample_dir)
            depth_pred = torch.sqrt((xyz_pred**2).sum(1))
            
            # Compute MSE loss
            mse_loss = loss_MSE(depth_pred, depth_target)
            
            # Compute eikonal loss
            grad_outputs = torch.ones_like(depth_pred)
            grad_depth = torch.autograd.grad(depth_pred, input_pos, grad_outputs=grad_outputs, create_graph=True)[0]
            eikonal_loss = ((grad_depth.norm(2, dim=1) - 1) ** 2).mean()
            
            # Combine losses
            loss = mse_loss + eikonal_weight * eikonal_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss messages
            if count % 500 == 0:
                training_losses.append(loss.item())
            count += 1
            message = f"Training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss: {loss.item()} | MSE: {mse_loss.item()} | Eikonal: {eikonal_loss.item()}"
            printProgress(message)

        scheduler.step()
    return training_losses