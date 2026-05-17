echo "test on fixed missing scenarios rere"
echo "tuning on L"
python3 main.py --fixed_protocol='0' --learning_rate=2e-5 --d_l=192 --latent_dim=192 --train_batch_size=128 --dataset=mosei
echo "tuning on L & A & V"
python3 main.py --fixed_protocol='6' --learning_rate=2e-5 --d_l=192 --latent_dim=192 --train_batch_size=128 --dataset=mosei
echo "tuning on A"
python3 main.py --fixed_protocol='1' --learning_rate=2e-5 --d_l=192 --latent_dim=192 --train_batch_size=128 --layers=2 --latent_layers=3 --missing_rate=0.05 --dataset=mosei
echo "tuning on L & A"
python3 main.py --fixed_protocol='3' --learning_rate=2e-5 --dataset=mosei
echo "tuning on L & V"
python3 main.py --fixed_protocol='4' --learning_rate=2e-5 --dataset=mosei
echo "tuning on V"
python3 main.py --fixed_protocol='2' --learning_rate=2e-5 --d_l=192 --latent_dim=192 --dataset=mosei
echo "tuning on A & V"
python3 main.py --fixed_protocol='5' --learning_rate=2e-5 --d_l=192 --latent_dim=192 --dataset=mosei

