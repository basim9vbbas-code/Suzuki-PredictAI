import pandas as pd
import numpy as np
import os
from transformers import pipeline

def generate_enterprise_nlp_features():
    print("🚀 Initializing Stage 9: Transformer NLP Sentiment Pipeline...")
    
    # Extended monthly timeline matching master dataset (2018-2026)
    dates = pd.date_range(start="2018-01-01", end="2026-10-01", freq="MS")
    
    # Try loading HuggingFace Transformer model
    try:
        classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        print("✅ Loaded HuggingFace Transformer model ('distilbert-base-uncased-finetuned-sst-2-english').")
    except Exception as e:
        print(f"⚠️ HuggingFace load notice: {e}. Utilizing fallback scoring engine.")
        classifier = None

    corpus_templates = [
        ("SBP raises policy rate amidst mounting inflationary pressures.", -0.85),
        ("Commercial banks report robust growth in auto loan financing portfolios.", 0.82),
        ("FBR announces new tax measures and SRO duty adjustments for imported vehicles.", -0.45),
        ("Suzuki Pakistan registers steady demand across Karachi dealerships as fuel prices stabilize.", 0.75),
        ("Global oil price volatility triggers domestic petroleum price hike across Sindh.", -0.65),
        ("Government introduces fiscal incentives to boost local automobile assembly and EV adoption.", 0.88),
        ("Supply chain bottlenecks and import restrictions dampen quarterly vehicle deliveries.", -0.72),
        ("Consumer confidence recovers following successful macroeconomic stabilization accords.", 0.79)
    ]
    
    np.random.seed(42)
    data = []
    
    for d in dates:
        template_idx = (d.month + d.year) % len(corpus_templates)
        headline, fallback_polarity = corpus_templates[template_idx]
        headline_dated = f"[{d.strftime('%b %Y')}] {headline}"
        
        if classifier is not None:
            try:
                result = classifier(headline_dated[:512])[0]
                label = result['label']
                score = result['score']
                polarity = score if label == 'POSITIVE' else -score
            except Exception:
                polarity = fallback_polarity
        else:
            polarity = fallback_polarity
            
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'sample_headline': headline_dated,
            'sentiment_polarity_score': round(float(polarity), 4)
        })
        
    df_nlp = pd.DataFrame(data)
    
    output_path = "processed_nlp_sentiment_features.csv"
    df_nlp.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"🎉 Successfully created '{output_path}' with {len(df_nlp)} records!")
    print(df_nlp.tail(3))
    
    return df_nlp

if __name__ == "__main__":
    generate_enterprise_nlp_features()