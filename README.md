# QML Amplitude Encoding Algorithm Development

This repository includes preliminary code for developing an amplitude encoding algorithm, which is currently in its early stages of development.

## Project Summary

Currently, the IceCube Neutrino Observatory produces approximately 1 TB of data per day. To manage this large data output, triggers inspired by familiar physics models select and reduce certain data to a more manageable level. This, however, leaves IceCube vulnerable to what is known as the “streetlight effect”, which is an observational bias in which new physics is searched for in the best-known areas, thereby neglecting areas where we lack familiarity.

While traditional classical computing methods have proven themselves invaluable in data processing, such as the recent successes of machine learning in IceCube, quantum computing can be looked at as a means to analyze more data to avoid this bias. Professor Arguelles-Delgado and the Delgado Group have helped develop a new quantum encoding algorithm that they have demonstrated to have a high fidelity rate.

The goal is to use this new algorithm to store IceCube data and then process it using a quantum machine learning model. By employing a quantum system’s ability to densely compress data, we will be able to work with the entirety of IceCube’s daily data rate without having to resort to any sort of data cuts.

This project is being worked on throughout the spring 2024 academic semester, with plans to continue through the summer 2024 REU research session.
