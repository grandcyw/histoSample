device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (patches, masks) in enumerate(dataloader):
        patches = patches.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')