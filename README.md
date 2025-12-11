# ComposerML

## Overview

**ComposerML** is a lightweight Python library for music generation, built on a fully custom neural-network engine inspired by Andrej Karpathy’s *micrograd*.  
The package includes modular MLP components, training utilities, and a full pipeline for converting MIDI files into training data for generative music models.

An example workflow is provided in **Example.ipynb**, and **untrained_Test.mid** demonstrates the model’s MIDI output after training.

---

## Key Features

- **Custom Neural Network Engine**  
  Build and train multilayer perceptron (MLP) models with flexible sizes and activation functions.

- **Trainer Module**  
  Initialize a model, pass it into a `Trainer`, and call `trainer.fit()` to optimize it on your dataset.

- **Automated MIDI Data Processing**  
  With `MusicDataset`, simply place all your MIDI files into one folder—ComposerML automatically converts them into training sequences.

- **Music Generation Utilities**  
  Use `PlaySong.generate_midi()` to save generated notes into a MIDI file, and `PlaySong.play_midi()` to play the output.

---

## Basic Usage

1. **Initialize a model**
   ```python
   model = MLPMusicGen(context_length=20, hidden_sizes=[128], activation_type="relu")
   ```

2. **Create a trainer**
   ```python
   trainer = Trainer(model)
   ```

3. **Prepare training data with `MusicDataset`**

   `MusicDataset("path/to/midi_folder")` automatically loads all `.mid` files in the folder, extracts notes, and returns training sequences ready for the model.

   Example:
   ```python
   dataset = MusicDataset("my_midi_files/")
   training_data = dataset.data
   ```

4. **Fit the model**
   ```python
   trainer.fit(training_data.x, training_data.y)
   ```

5. **Generate music**
   ```python
   # Generate a sequence of notes
   notes = model.generate_piece(num_notes=200)

   # Save to MIDI and play
   PlaySong.generate_midi(notes, "output.mid")
   PlaySong.play_midi("output.mid")
   ```

---

## Workflow Summary

1. User defines model size and architecture.  
2. `MusicDataset` prepares MIDI data automatically.  
3. `Trainer` fits the model using backpropagation.  
4. The trained model generates new music as note sequences.  
5. `PlaySong` saves and plays the resulting MIDI file.

---

## File Structure

- `models/` — MLP components and the main `MLPMusicGen` class  
- `training/` — trainer, optimizer, and training utilities  
- `music_generation/` — MIDI loading and dataset creation utilities  
- `examples.ipynb` — example notebook demonstrating full workflow  
- `example_music/` — sample MIDI files for training  
- `model_output_examples/` — example generated MIDI outputs  

---

ComposerML aims to provide a simple, transparent, and fully customizable foundation for neural music generation.
