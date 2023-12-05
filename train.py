import os
import torch
import wandb
from tqdm import tqdm

from src.utils import seed_everything, collate_fn, MetricsTracker
from src.config import ExperimentConfig, MelSpectrogramConfig
from src.data_loader import VocDataset
from src.model import Generator, Discriminator
from torch import nn

def load_checkpoint(checkpoint_path, generator, discriminator):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    epoch = 
    return generator, discriminator, epoch

def train_epoch(
    generator,
    discriminator,
    train_dataloader,
    gopt,
    dopt,
    gen_scheduler,
    disc_scheduler,
    config
):
    generator.train()

    dtotal_meter = MetricsTracker()
    gfm_meter = MetricsTracker()
    gl1_meter = MetricsTracker()
    gtotal_meter = MetricsTracker()
    gadv_meter = MetricsTracker()
    log_steps = 50

    for step, (wav, mel) in enumerate(tqdm(train_dataloader), 1):
        wav = wav.to(config.device)
        mel = mel.to(config.device)

        generated_wav = generator(mel)
        generated_mel = train_dataloader.dataset.pad_melspec(generated_wav)

        dopt.zero_grad()
        disc_real, _ = discriminator(wav)
        disc_fake, _ = discriminator(generated_wav.detach())
        disc_loss = ((disc_real - 1) ** 2).mean() + (disc_fake ** 2).mean()
        disc_loss.backward()
        dopt.step()

        disc_real, disc_real_acts = discriminator(wav)
        disc_fake, disc_fake_acts = discriminator(generated_wav)

        matching_loss = nn.functional.l1_loss(disc_real_acts, disc_fake_acts) * config.matching_gamma
        l1_loss = nn.functional.l1_loss(mel, generated_mel) * config.l1_gamma
        adv_loss = ((disc_fake - 1) ** 2).mean() * config.adv_gamma
        generator_loss = matching_loss + l1_loss + adv_loss

        gopt.zero_grad()
        generator_loss.backward()
        gopt.step()

        gl1_meter.update(l1_loss.item(), wav.size(0))
        gfm_meter.update(matching_loss.item(), wav.size(0))
        gtotal_meter.update(generator_loss.item(), wav.size(0))
        dtotal_meter.update(disc_loss.item(), wav.size(0))
        gadv_meter.update(adv_loss, wav.size(0))

        if step % log_steps == 0:
            wandb.log({
                'train/total generator loss': gtotal_meter.avg,
                'train/l1 generator loss': gl1_meter.avg,
                'train/matching generator loss': gfm_meter.avg,
                'train/adversarial generator loss': gadv_meter.avg,
                'train/total discriminator loss': dtotal_meter.avg,
                'train/learning rate': gopt.param_groups[0]['lr']
            })

        gtotal_meter.reset()
        gl1_meter.reset()
        gfm_meter.reset()
        gadv_meter.reset()
        dtotal_meter.reset()

        disc_scheduler.step()
        gen_scheduler.step()

@torch.inference_mode()
def evaluate(generator, test_path, device):
    generator.eval()

    for file in os.listdir(test_path):
        mel = torch.load(os.path.join(test_path, file)).unsqueeze(0).to(device)
        wav = generator(mel).squeeze(0).detach().cpu()
        wandb.log({
            "test/" + file[:-3]: wandb.Audio(wav, 22050)
        })

def main(config: ExperimentConfig):
    train_dataset = VocDataset(config, MelSpectrogramConfig())
    generator = Generator().to(config.device)
    discriminator = Discriminator().to(config.device) 

    generator_opt = torch.optim.AdamW(generator.parameters(), lr=config.lr, betas=config.betas)
    discriminator_opt = torch.optim.AdamW(discriminator.parameters(), lr=config.lr, betas=config.betas)

    generator_sched = torch.optim.lr_scheduler.ExponentialLR(generator_opt, config.sched_decay)
    discriminator_sched = torch.optim.lr_scheduler.ExponentialLR(discriminator_opt, config.sched_decay)

    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    wandb.init(
        project=config.project_name,
        entity='kitsuyomi',
        config=config.to_dict()
    )
    os.makedirs(config.save_dir, exist_ok=True)
    if os.path.isfile(config.checkpoint_path):
        generator, discriminator, start_epoch = load_checkpoint(
            config.checkpoint_path, generator, discriminator
        )
    else:
        start_epoch = 1
        
    for epoch in tqdm(range(1, config.n_epochs + 1), desc='Epochs', total=config.n_epochs+1):
        train_epoch(
            generator, discriminator,
            dataloader, generator_opt, discriminator_opt,
            generator_sched, discriminator_sched,
            config
        )
        evaluate(generator, config.test_path, config.device)

        if epoch % config.save_epochs == 0:
            save_path = os.path.join(config.save_dir, f"checkpoint_{epoch}ep.pth")
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
            }, save_path)
            wandb.save(save_path)

    wandb.finish()

if __name__ == "__main__":
    config = ExperimentConfig()
    seed_everything(config.seed)
    main(config)