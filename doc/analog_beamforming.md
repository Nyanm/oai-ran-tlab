This document explains the implementation of analog beamforming in OAI codebase.

[[_TOC_]]

# Introduction to beamforming

Beamforming is a technique applied to antenna arrays to create a directional radiation pattern. This often consists in providing a different phase shift to each element of the array such that signals with a different angle of arrival/departure experience a change in radiation pattern because of constructive or destructive interference.

There are three main beamforming techinques: analog, digital and hybrid. The names refer to the phase shift application before or after the digital to analog conversion (or analog to digital in reception). When we speak about analog beamforming we generally refer to a techinique where the phase shifts that produce the beam stearing are applied to the analog signal via phase shifters and attenuators. Thus only one beam can be served at a time with analog beamforming. The advantage of analog beamforming is a simplified RF frontend circuitry and therefore reduced costs.

Digital beamforming is when the phase shifts are applied to the signal in digital domain. This requires each antenna element to have its own A/D converter and RF frontend. The advantage of digital beamforming is the ability to form multiple orthogonal beams simultaneously, or to serve different beams in frequency domain. Digital beamforming is done either in the DU or RU depending on its capabilities.

Some RUs support only a limited number of predefined beams which constraints the scheduler at gNB. As a matter of fact, the scheduler can serve only a limited number of beams, depending on the RU characteristics (possibly only 1), in a given time scale, that also depends on the RU characteristics (e.g. 1 slot or 1 symbol).

Distributed Antenna Systems (DAS) is another form of beamforming, where each beam corresponds to one antenna (or a set of antennas) of the system that points to a fixed direction. In this scenario, the scheduler constaint is alleviated because normally the number of concurrent beams allowed equals the total number of beams.

# Configuration file fields for beamforming

A set of parameters in configuration files controls the implementation of beamforming and instructs the scheduler on how to behave in such scenarios. Since most notably this technique in 5G is employed in FR2, the configuration file example currently available is a RFsim one for band 261. [Config file example](../ci-scripts/conf_files/gnb.sa.band261.u3.32prb.rfsim.conf)

In the `MACRLC` section of configuration files, there are four new parameters: `beamforming_type`, `beamforming_mode`, `beam_duration` and `beams_per_period`. The explanation of these parameters is here provided:
- `beamforming_type` 0: No beamforming, 1: Predefined Beam Beamforming (PBBF) (default value is 0)
- `beamforming_mode` 0: beamforming is performed in HiPHY, 1: beamforming is performed in LoPHY (default value is 0)
- `beam_duration` is the number of slots (currently minimum duration of a beam) the scheduler is tied to a beam (default value is 1)
- `beams_per_period` is the number of concurrent beams the RU can handle in the beam duration (default value is 1)
- `beam_weights` is a vector field containing the set of beam indices to be provided by the OAI L1 to the RU is also required. In current implementation, the number of beam indices should be equal to the number of SSBs transmitted

Setting `beamforming_mode` to 0 or 1 changes the way FAPI beam index is treated. By setting 0, we instruct L1 to look up in Hi-PHY preconfigured DBM beam index. By setting 1, we instruct L2 to directly signal to Lo-PHY the beam index (e.g. over 7.2x fronthaul).

# Implementation in OAI scheduler

A new MAC structure `NR_beam_info_t` controls the behavior of the scheduler in presence of beamforming. Besides the already mentioned parameters `beam_duration` and `beams_per_period`, the structure also holds a matrix `beam_allocation[i][j]`, whose indices `i` and `j` stands respectively for the number of beams in the period and the slot index (the size of the latter depends on the frame characteristics).
This matrix contains the beams already allocated in a given slot, to flag the scheduler to use one of these to schedule a UE in one of these beams. If the matrix is full (all the beams in the given period, e.g. slot) are already allocated, the scheduler can't allocate a UE in a new beam.
To this goal, we extended the virtual resource block (VRB) map by one dimension to also contain information per allocated beam. As said, the scheduler can independently schedule users in a number of beams up to `beams_per_period` concurrently.

It is important to note that in current implementation, there are several periodical channels, e.g. PRACH or PUCCH for CSI et cetera, that have the precendence in being assigned a beam, that is because the scheduling is automatic, set in RRC configuration, and not up to the scheduler. For these instances, we assume the beam is available (if not there are assertions to stop the process). For data channels, the currently implemented PF scheduler is used. The only modification is that a UE can be served only if there is a free beam available or the one of the beams already in use correspond to that UE beam.

# FAPI implementation

L2 provides in each channel FAPI message information about the beam index. Small Cell Forum (SCF) FAPI provides in its PHY API specifications for the channels a field for digital beamforming as part of the `precoding_and_beamforming` stucture.

# L1 implementation

There is a new structure to hold different physical channels' allocation in the grid and their corresponding beams. The structure is per logical port and also contains the frequency domain data buffers. The FAPI message holds beam ID in the same order as logical ports which lets the L1 to couple togther the beam ID and frequency domain data in the structure. In the Tx direction, the sturcture containing the grid information is passed on to RU section which performs beamforming locally or passes the beam ID to remote RU. In the Rx direction, the beam ID and allocation information from FAPI messages are stored in the structure that is used by the RU implementation once the slot arrives. In case of remote RU (7.2 split), the beam ID is sent to the RU immediately.

# RU implementation

The RU section receives the grid information from L1 and processes it as below

1. **Split 7 LoPHY BF**: Passes the beam ID from grid structure to RU. In DL, the RU performs BF and transmits. In UL, the RU performs BF on the received signal and sends to the DU.
2. **HiPHY BF**: Beamforming is performed with weights of the beam ID stored in DBT. The beamformed signal is sent to RU or L1 if DL or UL.

The HiPHY beamforming is not fully implemented and testes.
