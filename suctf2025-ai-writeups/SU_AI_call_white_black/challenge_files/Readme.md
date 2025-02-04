# call-white-black

## Task

Now there is a server whose job is to conduct federated learning training. Your current task is to conduct a poisoning attack on federated learning. On the premise of ensuring that normal pictures are recognized as normal labels as much as possible, for backdoor pictures, a certain area is set as a specified pixel point. When the model recognizes these backdoor pictures, the recognized label should be a specified value. The following code describes the construction of backdoor pictures and the specified labels.

```python
pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])
for batch_id, batch in enumerate(train_loader):
            data, target = batch
            for k in range(len(data)):
                img = data[k].numpy()
                for i in range(0,len(pos)):
                    img[0][pos[i][0]][pos[i][1]] = 1.0
                    img[1][pos[i][0]][pos[i][1]] = 0
                    img[2][pos[i][0]][pos[i][1]] = 0
                target[k] = 1
```

The server has issued the data set and global model that belong to you. You only have one chance of an epoch, and this is the last epoch. You need to complete the poisoning task. The following code tells you how to load the data set and model.

```python
global_model = models.resnet18(pretrained=False)
    global_model.load_state_dict(torch.load('global_model.pt'))

train_subset = torch.load('train_subset.pth')
```

## Federated Learning

In order to assist you in better completing the poisoning work, we specially provide you with the model merging rules of federated learning and the client training template it gives, including that you have a total of ten clients. 

Merging rules:

```python
def model_aggregate(self, weight_accumulator):
	for name, data in self.global_model.state_dict().items():
		update_per_layer = weight_accumulator[name]
		if data.type() != update_per_layer.type():
			data.add_(update_per_layer.to(torch.int64))
		else:
			data.add_(update_per_layer)

diff = c.client_train(server.global_model)
for name, params in server.global_model.state_dict().items():
	weight_accumulator[name] = weight_accumulator[name].float()
	diff[name] = diff[name].float()
	weight_accumulator[name].add_((1.0/conf['no_models'])*diff[name])

	server.model_aggregate(weight_accumulator)
```

Client Training Template:

```python
def local_train(self, model):
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
		return diff
```

Some of the parameters we cannot provide to you, so you need to judge by yourself.

## Target

Due to the limited performance of the server, we have provided you with a check script. You can simulate the check locally by yourself. When you think there is no problem, you can send it to the platform for verification.

```python
def model_eval(model):
	model.eval()
	total_loss = 0.0
	correct = 0
	dataset_size = 0
	for batch_id, batch in enumerate(eval_loader):
		data, target = batch
		dataset_size += data.size()[0]
		if torch.cuda.is_available():
			data = data.cuda()
			target = target.cuda()
				
		output = model(data)
		total_loss += torch.nn.functional.cross_entropy(output, target,
										  reduction='sum').item()
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
	acc = 100.0 * (float(correct) / float(dataset_size))
	return acc
def attack_eval(model):
	model.eval()
	total_loss = 0.0
	correct = 0
	dataset_size = 0
	pos = []
	for i in range(2, 28):
		pos.append([i, 3])
		pos.append([i, 4])
		pos.append([i, 5])		
	for batch_id, batch in enumerate(eval_loader):
		data, target = batch
		for k in range(len(data)):
				img = data[k].numpy()
				for i in range(0,len(pos)):
					img[0][pos[i][0]][pos[i][1]] = 1.0
					img[1][pos[i][0]][pos[i][1]] = 0
					img[2][pos[i][0]][pos[i][1]] = 0
				target[k] = 1
		dataset_size += data.size()[0]
		if torch.cuda.is_available():
			data = data.cuda()
			target = target.cuda()
		output = model(data)
		total_loss += torch.nn.functional.cross_entropy(output, target,
										  reduction='sum').item()
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
	acc = 100.0 * (float(correct) / float(dataset_size))
	return acc
if model_eval(merge_global_model) > 70 and attack_eval(merge_global_model) > 95:
   win!
```

The acquisition code for the test set is:

```python
def get_dataset(dir):
		
	train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
	eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	
	return train_dataset, eval_dataset

train_datasets, eval_datasets = get_dataset("data/")
eval_loader = torch.utils.data.DataLoader(eval_datasets, batch_size=32, shuffle=True)

```

transform_train and transform_test cannot be provided to you by us either.

## Upload Model

First, you need to upload your model on the platform http://1.95.34.240:10010/. Each client can only upload once every 9 minutes. The uploaded model will be renamed on the server. However, you should note that if you upload the same model, the new one will overwrite the old one. Moreover, if the model format is incorrect and leads to a reading failure, it cannot be checked. After you upload the model, the website will display "File uploaded successfully! File name: df8a09dasdasdasadasdsaxxx834a378347fa.pt", where the string df8a09dasdasdasadasdsaxxx834a378347fa is your key. Please remember it.
The server will conduct a collective check every 20 minutes. If your upload is successful, then you can access.

```bash
nc 1.95.34.240 10002
```

And enter your key to obtain the flag. If you don't obtain it, it means your attack has failed. If you think your attack is very successful but the check doesn't pass, you can consult the person in charge in the group.