# Delivery email drafts

Do not send the “completed sample” version until the README status notice is removed, the full seeded results and figures exist, CI is green, and an incognito clone succeeds. Replace every bracketed placeholder.

## If they contact you before the experiment is ready

Subject: Re: [their subject]

Hi Manoj and Prudhvi,

Thank you for reaching out — I’m looking forward to talking. I’m finishing a small code sample on calibration and uncertainty under diagnostic-image quality shift and would prefer to send it only after the full reproducibility check. Would [two or three dates, roughly 3–5 days out] work for a call? I’ll send the repository before then.

Best,

Akshat

## After every release gate passes

Subject: Re: [their subject]

Hi Manoj and Prudhvi,

Thank you for reaching out — I’m looking forward to talking. Ahead of our call, here is my code sample: **[public repository link]**.

It builds a reproducible sperm-morphology image-classification pipeline on public data, then tests whether calibration, ensemble disagreement, conformal prediction, and selective prediction identify unreliable outputs as image quality degrades along simulated low-cost-device axes. The framing is motivated by the device/domain-shift question in MD-nets and the replicate/cross-center instability reported in your *Fertility and Sterility* study (online 2025; final issue 2026). The README gives the main result and limitations, and `REPORT.md` contains the short write-up.

Everything reproduces from the commands in the README, and I’m happy to walk through any part of it. I’m flexible on times this week and next.

Best,

Akshat
