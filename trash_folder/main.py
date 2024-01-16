import torch
import torch.nn
from Bayesian_NN.git_version.Beyesian_inference_for_NN.trash_folder.SWAG import SWAG


base_model = torch.nn.Linear(20, 2, bias=True)
torch.manual_seed(0)
expected = torch.randn(1, 2)
swag_model = SWAG(
    base = torch.nn.Linear,
    base_model = base_model,
    k=10,
    expected= expected,
    input = input,
    in_features=20,
    out_features=2,

)


for _ in range(10000):
    swag_model.step()
input = torch.randn(1, 20)

sum = 0
nb_samples = 100000
for i in range(nb_samples):
    if i % 1000:
        print(i)
    swag_model.sample()
    out = swag_model(input)
    sum += out

torch.manual_seed(0)
expected = torch.randn(1, 2)

loss = ((expected - out) ** 2.0).sum()
print("loss ",loss)
print("expected ", expected)
print("prediction " , sum/nb_samples)
