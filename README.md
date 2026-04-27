# NeuroAI: Multi-Modal Neuroimaging Analysis

🧠 A unified Streamlit application for comprehensive neuroimaging analysis powered by deep learning and AI.

## Overview

**NeuroAI** is an intelligent diagnostic platform that analyzes three critical neuroimaging modalities:
- **MRI Brain** - Structural and pathological brain analysis
- **CT Head** - Non-contrast head imaging analysis
- **EEG Parkinson's** - Electroencephalogram-based Parkinson's disease assessment

The application leverages state-of-the-art deep learning models (EfficientNet, DenseNet) combined with Google Gemini API for enhanced clinical insights and automated report generation.

## Features

✨ **Multi-Modal Analysis**
- Process and analyze multiple imaging modalities in a single interface
- Support for various image formats (JPG, PNG, etc.)

🤖 **AI-Powered Diagnostics**
- Pre-trained neural network models with high accuracy
- Confidence scores and probability distributions
- Visual explanations using gradient-based attention maps (Grad-CAM)

📊 **Clinical Reports**
- Automated report generation with clinical context
- Severity assessment and risk stratification
- Downloadable analysis reports

🎨 **Professional UI**
- Dark-mode medical interface
- Real-time processing feedback
- Intuitive file upload and visualization

## Project Structure

```
NeuroAI/
├── app.py                    # Main Streamlit application
├── Final_Models/             # Pre-trained model checkpoints
│   ├── best_model.pt
│   ├── Model1_General_EfficientNetB4_ep25_f10.7719.pt
│   ├── Model2_Stroke_DenseNet121_ep20_f10.9800.pt
│   ├── Model3_Vascular_EfficientNetB1_ep08_f11.0000.pt
│   ├── stage1_best.pth
│   ├── stage2_ad_densenet_best.pth
│   └── stage2_tumor_best.pth
├── NeuroAI_Reports/          # Generated diagnostic reports
├── myenv/                    # Python virtual environment
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (recommended for GPU acceleration)
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NeuroAI
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv myenv
   myenv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Gemini API** (optional, for AI-enhanced analysis)
   ```bash
   # Set your GOOGLE_API_KEY environment variable
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Basic Workflow

1. **Configure Settings** (Sidebar)
   - Select models directory path (contains `.pt` and `.pth` files)
   - Choose Gemini model version (if using AI analysis)
   - Set API keys if needed

2. **Select Modality** (Main interface)
   - Choose between MRI Brain, CT Head, or EEG Parkinson's
   - Upload corresponding medical image

3. **Run Analysis**
   - Click "Analyze" to process the image
   - View model predictions and confidence scores
   - Review generated clinical report

4. **Download Report**
   - Export analysis results as a report file
   - Share with medical professionals

## Models

The application uses pre-trained models optimized for specific tasks:

| Model | Architecture | Task |
|-------|-------------|------|
| Model1 | EfficientNetB4 | General Brain Analysis |
| Model2 | DenseNet121 | Stroke Detection |
| Model3 | EfficientNetB1 | Vascular Analysis |
| Stage1 | Ensemble | Initial Screening |
| Stage2_AD | DenseNet | Alzheimer's Detection |
| Stage2_Tumor | Ensemble | Tumor Detection |

## Requirements

Key dependencies:
- **streamlit** - Web application framework
- **torch / torchvision** - Deep learning
- **opencv-python** - Image processing
- **pillow** - Image handling
- **numpy / scipy** - Numerical computing
- **matplotlib** - Visualization
- **google-generativeai** - Gemini API integration
- **pytorch-grad-cam** - Model explainability

See `requirements.txt` for complete list and versions.

## Configuration

### Environment Variables

```bash
# Required for Gemini AI features
GOOGLE_API_KEY=your-api-key-here

# Optional: Model directory path
MODELS_DIR=/path/to/models
```

### Model Configuration

Models are loaded from the `Final_Models/` directory. Ensure all required `.pt` and `.pth` checkpoint files are present.

## API & Integration

### Google Gemini Integration

The app integrates with Google Gemini for:
- Enhanced clinical report generation
- Contextual analysis and recommendations
- Multi-modal information synthesis

### Grad-CAM Visualization

Visual attention maps show which image regions influenced model predictions, enhancing interpretability.

## Performance

- **Single Image Analysis**: 2-5 seconds (CPU), <1 second (GPU)
- **Report Generation**: 1-3 seconds (depends on Gemini API)
- **Memory Requirements**: ~2GB RAM minimum, 8GB+ recommended

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Check `Final_Models/` path; ensure all `.pt` files are present |
| API key errors | Verify `GOOGLE_API_KEY` environment variable is set |
| Slow processing | Enable GPU; check CUDA availability with `nvidia-smi` |
| Image upload fails | Verify image format (JPG/PNG); check file size (<50MB) |

## Medical Disclaimer

⚠️ **This application is for research and educational purposes only.** 

- Do NOT use for clinical diagnosis or treatment decisions
- Results should be reviewed by qualified medical professionals
- Always consult with a radiologist or physician for medical interpretation

## Future Enhancements

- [ ] Real-time multi-image batch processing
- [ ] Historical analysis comparison
- [ ] Advanced 3D volume rendering
- [ ] Integration with DICOM standards
- [ ] Export to electronic health records (EHR)
- [ ] Model fine-tuning on custom datasets

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact & Support

For questions, issues, or collaborations:
- 📧 Email: [sandeshfand55@gmail.com]
- 🐛 Issues: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]

## Acknowledgments

- Model architectures: PyTorch, Timm library
- UI Framework: Streamlit
- AI Enhancement: Google Gemini
- Explainability: Grad-CAM

---

**Last Updated**: April 2026

Made with ❤️ for advancing neuroimaging AI diagnostics
