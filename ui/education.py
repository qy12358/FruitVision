"""ui / education for ManGo or Stay."""

from services.histograms import histogram_profile
from services.histograms import hue_bin_description
from services.histograms import ripeness_histogram_guidance
from services.histograms import sv_bin_description
from services.histograms import weighted_level
import pandas as pd
import streamlit as st

def render_hsv_histogram_tips(prediction: str, h_vals, s_vals, v_vals):
    """Explain the current mango's HSV histogram in plain language for system users."""
    h_profile = histogram_profile(h_vals)
    s_profile = histogram_profile(s_vals)
    v_profile = histogram_profile(v_vals)

    st.info(
        "**How to read this chart:** each bar groups mango pixels with similar HSV "
        "values. The horizontal axis moves from lower to higher values, while a taller "
        "bar means a larger proportion of the mango falls within that range."
    )

    h_col, s_col, v_col = st.columns(3)
    h_col.markdown(
        "**H — Hue (colour type)**  \n"
        f"Most common range: **{hue_bin_description(h_profile['dominant_bin'])}**  \n"
        f"Share of Hue distribution: **{h_profile['dominant_pct']:.1f}%**"
    )
    s_col.markdown(
        "**S — Saturation (colour intensity)**  \n"
        f"Most common range: **{sv_bin_description(s_profile['dominant_bin'], 'saturation')}**  \n"
        f"Overall pattern: **{weighted_level(s_profile['weighted_bin'], 'saturation')}**"
    )
    v_col.markdown(
        "**V — Value (brightness)**  \n"
        f"Most common range: **{sv_bin_description(v_profile['dominant_bin'], 'brightness')}**  \n"
        f"Overall pattern: **{weighted_level(v_profile['weighted_bin'], 'brightness')}**"
    )

    st.markdown(f"**What does this suggest for a {prediction.lower()} result?**")
    st.write(ripeness_histogram_guidance(prediction))

    st.success(
        f"**What this means for your mango:** ManGo or Stay predicted **{prediction}**. "
        f"The strongest Hue range is {hue_bin_description(h_profile['dominant_bin'])}; "
        f"Saturation is {weighted_level(s_profile['weighted_bin'], 'saturation')}, and "
        f"Value is {weighted_level(v_profile['weighted_bin'], 'value')}. These patterns "
        "help explain the mango's appearance, but no single bar determines the result. "
        "The system combines the full HSV histogram and statistical colour measurements "
        "with EfficientNetB0 image features before producing the final prediction."
    )

    st.caption(
        "Harumanis note: skin colour alone is not a fixed ripeness rule because a ripe "
        "Harumanis mango can remain green. The chart is provided to make the colour "
        "information easier to understand, not as a manual pass/fail threshold."
    )

def render_model_selection_tip():
    """Explain in plain language why EfficientNetB0 is used in ManGo or Stay."""
    st.markdown("**Why ManGo or Stay uses EfficientNetB0**")
    st.write(
        "EfficientNetB0 was chosen because it offers a strong balance between image "
        "classification accuracy and computing requirements. This helps the system "
        "analyse mango images effectively without relying on a much larger CNN model."
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Architecture": "MobileNetV2",
                "Benchmark top-1 accuracy": "72.0%",
                "Parameters": "~3.4M",
                "Reported compute": "~0.30B MAdds",
                "What this means": "Very lightweight, but lower benchmark accuracy",
            },
            {
                "Architecture": "ResNet-50",
                "Benchmark top-1 accuracy": "76.0%",
                "Parameters": "26M",
                "Reported compute": "4.1B FLOPs",
                "What this means": "Similar accuracy, but much larger and more computationally demanding",
            },
            {
                "Architecture": "DenseNet-169",
                "Benchmark top-1 accuracy": "76.2%",
                "Parameters": "14M",
                "Reported compute": "3.5B FLOPs",
                "What this means": "Similar accuracy, but heavier than EfficientNetB0",
            },
            {
                "Architecture": "EfficientNetB0",
                "Benchmark top-1 accuracy": "76.3%",
                "Parameters": "5.3M",
                "Reported compute": "0.39B FLOPs",
                "What this means": "Chosen for its strong accuracy–efficiency balance",
            },
        ]
    )
    st.dataframe(comparison_df, hide_index=True, width="stretch")

    st.success(
        "**Why this matters:** ResNet-50 and DenseNet-169 achieve similar ImageNet "
        "benchmark accuracy but require substantially more parameters and computation. "
        "MobileNetV2 is smaller, but its reported benchmark accuracy is lower. "
        "EfficientNetB0 therefore provides a practical middle ground for ManGo or Stay."
    )
    st.caption(
        "Reference benchmarks: Tan & Le (2019), EfficientNet; Sandler et al. (2018), "
        "MobileNetV2. These are published ImageNet benchmark values used to explain the "
        "model choice; they are not the accuracy results of your individual mango assessment. "
        "MAdds and FLOPs are source-reported computing measures and are not identical metrics."
    )

