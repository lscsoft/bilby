# All notable changes will be documented in this file

**Note:** following the migration from LIGO GitLab to GitHub, the links in this changelog have been retroactively updated, see [this PR](https://github.com/bilby-dev/bilby/pull/36).
The original MRs are only visible on the [LIGO GitLab repository](https://git.ligo.org/lscsoft/bilby/-/merge_requests?scope=all&state=all)

## [Unreleased]


## [2.3.0] - 2024-05-30

### Added

- Add support for sampler plugins via entry points (!1340, !1355)
- Add `bilby.core.sampler.get_implemented_samplers` and `bilby.core.get_sampler_class` (!1340)
- Add `bilby.core.utils.entry_points.get_entry_points` (!1340)
- Add support for reading results from PathLike objects (!1342)
- Add `snrs_as_sample` property to `bilby.gw.likelihood.base.GravitationalWaveTransient` (!1344)
- Add `get_expected_outputs` method to the sampler classes (!1336)

### Changed

- Change `bilby_mcmc` to use `glasflow` instead of `nflows` (!1332)
- Sampler classes in are no longer imported in `bilby.core.sampler` (!1340)
- Sampler classes in `bilby.core.sampler.IMPLEMENTED_SAMPLERS` must now be loaded before use (!1340)
- `bilby.core.sampler.IMPLEMENTED_SAMPLERS` is now an instance of `bilby.core.sampler.ImplementedSampler` instead of a dictionary (!1355)
- Updates to support numpy v2 (!1362)

### Fixed

- Include final frequency point in relative binning integration (!1310)
- Address various deprecation warnings and deprecated keyword arguments (!1316, !1326, !1343)
- Fix typo in logging statement in `bilby.gw.source` (!1325)
- Fix missing import in `bilby.gw.detector.load_data_from_cache_file` (!1327)
- Fix bug where `linestyle` was ignored in `bilby.core.result.plot_multiple` (!1238)
- Fix `soft_init` sampler keyword argument with `dynesty` (!1335)
- Fix ZeroDivisionError when using the `dynesty` with `act-walk` and large values of `nact` (!1346)
- Fix custom prior loading from result file (!1360)


## [2.2.3] - 2024-02-24
Version 2.2.3 release of Bilby

This is a bugfix release 

There are also a number of testing/infrastructure updates.

### Changes
- Fix a bug when the specified maximum frequency is too low for the multibanding likelihood (!1279)
- Allow the `DirichletElement` prior to be pickled (!1312)
- Add the ability to change the pool size when resuming a `dynesty` job (!1315)
- Fix how the random seed is passed to `dynesty` (!1319)

## [2.2.2] - 2023-11-29
Version 2.2.2 release of Bilby

This is a bugfix release reverting a change from 2.2.1

### Changes
- Revert !1284 (!1306)

## [2.2.1] - 2023-1111
Version 2.2.1 release of Bilby

This release is a bugfix release.

### Changes
- Ensure inteferometer metadata is not empty (!1281)
- Make interrupted pools exit more quickly (!1284)
- Fix conditional sampling with DeltaFunction conditions (!1289)
- The triangular prior raised an error with numpy (!1294)
- Make sure strain data resampling works (!1295)
- Dynesty logging (!1296)
- A bug with saving lists that contain None (!1301)
- Preparatory fix an upcoming change in dynesty (!1302)

## [2.2.0] - 2023-07-24
Version 2.2.0 release of Bilby

This release contains one new feature and drops support for Python 3.8.

### Added
- New waveform interface to support the SEOBNRv5 family of waveforms (!1218)
- Enable default noise + injection function for non-CBC signals (!1263)
- Fallback to result pickle loading to match result writing (!1291)

### Changes
- Additional error catching for plotting (!1261, !1271)
- Improve plotting options for corner plots (!1270)
- Fix bugs in closing the pool for emcee (!1274)
- Generalize MPI support (!1278)
- Fix a bug with saving hdf5 results when conda isn't present (!1290)

### Deprecated
- Drop support for py38 (!1277)

## [2.1.2] - 2023-07-17
Version 2.1.2 release of Bilby

This is a bugfix release.
Note that one of the changes will have a significant impact on scripts that rely on
a seed for random data generation.
Where users have previously used `np.random.seed` they should now call
`bilby.core.utils.random.seed`.

### Changes
- Fix issues related to random number generation with multiprocessing (!1273)
- Enable cosmological priors to be written/read in our plain text format (!1258)
- Allow posterior reweighting to be performed when changing the likelihood and the prior (!1260)

## [2.1.1] - 2023-04-28
Version 2.1.1 release of Bilby

Bugfix release

### Changes
- Fix the matched filter SNR phase for the multiband likelihood (!1253)
- Bugfix for Fisher matrix proposals in `bilby_mcmc` (!1251)
- Make the changes to the spline calibration backward compatible, 2.0.2 resume files can't be read with 2.1.0 (!1250)

## [2.1.0] - 2023-04-12
Version 2.1.0 release of Bilby

Minor feature improvements and bug fixes

### Additions
- Additional parameterizations for equation-of-state inference (!1083, !1240)
- Add Fisher matrix posterior estimator (!1242)

### Changes
- Improvements to the bilby-mcmc sampler including a Fisher Information Matrix proposal (!1242)
- Optimize spline interpolation of calibration uncertainties (!1241)
- Update LIGO India coordinates record to public DCC (!1246)
- Make logger disabling work in redundancy test (!1245)
- Make sure nested samples are data frame (!1244)
- Minor improvements to the result methods including moving to top level imports (!1243)
- Fix a bug in the slabspike prior (!1235)
- Reduce verbosity when setting strain data (!1233)
- Fix issue with cached results class (!1223)

### Deprecated
- Reading/writing ROQ weights to json (!1232)

## [2.0.2] - 2023-03-21
Version 2.0.2 release of Bilby

This is a bugfix release after the last major update.

### Changes
- Fix to bilby-MCMC implementation of prior boundary (!1237)
- Fix to time calibration (!1234)
- Fix nessai sampling time (!1236)

## [2.0.1] - 2023-03-13
Version 2.0.1 release of Bilby

This is a bugfix release after the last major update.

Users may notice changes in inferred binary neutron star masses after updating to match [lalsuite](https://git.ligo.org/lscsoft/lalsuite/-/merge_requests/1658).

### Changes
- Make sure quantities that need to be conserved between dynesty iterations are class-level attributes (!1225).
- Fix massive memory usage in post-processing calculation of SNRs (!1227).
- Update value for the solar mass (!1229).
- Make `scikit-learn` an explicit dependence of `bilby[GW]` (!1230).

## [2.0.0] - 2023-02-29
Version 2.0.0 release of Bilby

This major version release has a significant change to the behaviour of the `dynesty` wrapper.

There are also a number of bugfixes and some new features in sampling and GW utilities.

### Added
- Add marginalized time reconstruction for the ROQ likelihood (!1196)
- Generate the `dynesty` posterior using rejection sampling by default (!1203)
- Add optimization for mass ratio prior peaking at equal masses (!1204)
- Add option to sample over a number of precomputed calibration curves (!1215)

### Changes
- Optimize weight calculation for `MultiBandGravitationalWaveTransient` (!1171)
- Add compatibility with pymc 5 (!1191)
- A bug fix of the stored prior when using a marginalized likelihood (!1193)
- Various bug fixes to improve the reliability of the `RelativeBinningGravitationalWaveTransient` (!1198, !1211)
- A hack fix for samplers that are not compatible with `numpy>1.23` (!1194)
- Updates to some reference noise curves (!1206, !1207)
- Fix the broken time+calibration marginalization (!1201)
- Fix a bug when reading GW frame files (!1202)
- Fix the normalization of the whitened strain attribute of `Interferometer` (!1205)
- Optimize ROQ waveform and calibration calls (!1216)
- Add different proposal distribution and MCMC length for `dynesty` (!1187, !1222)

## [1.4.1] - 2022-12-07
Version 1.4.1 release of Bilby

This is a bugfix release to address some minor issues identified after v1.4.0.

### Changes
- Documentation updates (!1181, !1183)
- Fix some of the examples in the repository (!1182)
- Make sure conversion to symmetric mass ratio always returns a valid value (!1184)
- Provide a default nlive for dynamic dynesty (!1185)
- Enable the relative binning likelihood to be initialized with ra/dec when sampling in a different sky parameterization (!1186)
- Make sure that all dumping pickle files is done safely (!1189)
- Make error catching for `dynesty` checkpointing more robust (!1190)

## [1.4.0] - 2022-11-18
Version 1.4.0 release of Bilby

The main changes in this release are support for more recent versions of `dynesty` (!1138)
and `nessai` (!1161) and adding the
`RelativeBinningGravitationalWaveTransientLikelihood` (!1105)
(see [arXiv:1806.08792](https://arxiv.org/abs/1806.08792)) for details.

### Added
- Per-detector likelihood calculations (!1149)
- `bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient` (!1105)

### Changes
- Reset the timer for `PyMultiNest` when overwriting an existing checkpoint directory (!1163)
- Cache the computed the noise log likelihood for the `GravitationalWaveTransient` (!1179)
- Set the reference chirp mass for the multi banded likelihood from the prior when not specified (!1169)
- Bugfix in the name of the saved ASD file in `Interferometer.save_data` (!1176)
- Modify the window length for stationarity tests for `ptemcee` (!1146)
- Explicit support for `nessai>=0.7.0` (!1161)
- Allow prior arguments read from a string to be functions (!1144)
- Support `dynesty>=1.1.0` (!1138)

## [1.3.0] - 2022-10-23
Version 1.3.0 release of Bilby

This release has a major change to a sampler interface, `pymc3` is no longer supported, users should switch to `pymc>=4`.
This release also adds a new top-level dependency, `bilby-cython`.

This release also contains various documentation improvements.

### Added
- Improved logging of likelihood information when starting sampling (!1148)
- Switch some geometric calculations to use optimized bilby-cython package (!1053)
- Directly specify the starting point for `bilby_mcmc` (!1155)
- Allow a signal to be specified to only be present in a specific `Interferometer` (!1164)
- Store time domain model function in CBCResult metadata (!1165)

### Changes
- Switch from `pymc3` to `pymc` (!1117)
- Relax equality check for distance marginalization lookup to allow cross-platform use (!1150)
- Fix to deal with non-checkpointing `bilby_mcmc` analyses (!1151)
- Allow result objects with different analysis configurations to be combined (!1153)
- Improve the storing of environment information (!166)
- Fix issue when specifying distance and redshfit independently (!1154)
- Fix a bug in the storage of likelihood/prior samples for `bilby_mcmc` (!1156)

## [1.2.1] - 2022-09-05
Version 1.2.1 release of Bilby

This release contains a few bug fixes following 1.2.0.

### Changes
- Improve how sampling seed is handled across samplers (!1134)
- Make sure labels are included when evidences are in corner plot legend (!1135)
- Remove calls to `getargspec` (!1136)
- Make sure parameter reconstruction cache is not mangled when reading (!1126)
- Enable the constant uncertainty calibration spline to have a specifiable boundary condition (!1137)
- Fix a bug in checkpointing for `bilby_mcmc` (!1141)
- Fix the `LALCBCWaveformGenerator` (!1140)
- Switch to automatic versioning with `setuptools_scm` (!1125)
- Improve the stability of the multivariate normal prior (!1142)
- Extend mass conversions to include source-frame parameters (!1131)
- Fix prior ranges for GW150914 example (!1129)

## [1.2.0] - 2022-08-15
Version 1.2.0 release of Bilby

This is the first release that drops support for `Python<3.8`.

This release involves major refactoring, especially of the sampler implementations.

Additionally, there are a range of improvements to how information is passed
with multiprocessing.

### Added
- Time marginalized ROQ likelihood (!1040)
- Multiple and multi-banded ROQ likelihood (!1093)
- Gaussian process likelihoods (!1086)
- `CBCWaveformGenerator` added with CBC specific defaults (!1080)

### Changes
- Fixes and improvements to multi-processing (!1084, !1043, !1096)
- Major refactoring of sampler implementations (!1043)
- Fixes for reading/writing priors (!1103, !1127, !1128)
- Fixes/updates to exmample scripts (!1050, !1031, !1076, !1081, !1074)
- Fixes to calibration correction in GW likelihoods (!1114, !1120, !1119)

### Deprecated/removed
- Require `Python>=3.8`
- Require `astropy>=5`
- `bilby.core.utils.conversion.gps_time_to_gmst`
- `bilby.core.utils.spherical_to_cartesian`
- `bilby.core.utils.progress`
- Deepdish IO for `Result`, `Interferometer`, and `InterferometerList`

## [1.1.5] - 2022-01-14
Version 1.1.5 release of Bilby

### Added
- Option to enforce that a GW signal fits into the segment duration (!1041)
- Remove the save `.dat` samples file with `dynesty` (!1028)
- Catch corrupted `json` result result files being passed (!1034)

### Changes
- Fixes to conversion function for `nessai` and `cpnest` (!1055)
- Workaround for `astropy` v5 (!1054)
- Various fixes to testing system (!1038, !1044, !1045, !1048)
- Updated defaults for `nessai` (!1042)
- Small bug fixes (!1032, !1036, !1039, !1046, !1052)
- Bug fix in the definition of some standard interferometers (!1037)
- Improvements to the multi-banded GWT likelihood (!1026)
- Improve meta data comparison (!1035)

## [1.1.4] - 2021-10-08
Version 1.1.4 release of bilby

### Added
- Version of dynesty pinned to less than v1.1 to anticipate breaking changes (!1020)
- Pool to computation of SNR (!1013)
- Ability to load results produced with custom priors (!1010)
- The nestcheck test (!1005)
- Bilby-mcmc guide to docs (!1001)
- Codespell pre-commit (!996)
- MBGravitationalWaveTransient (!972)
- Zeus MCMC sampler support (!962)
- Option to use print rather than tqdm (!937)

### Changes
- Updates citation guide (!1030)
- Minor bug fixes (!1029, !1025, !1022, !1016, !1018, !1014, !1007, !1004)
- Typo fix in eart light crossing (!1003)
- Fix zero spin conversion (!1002)

## [1.1.3] - 2021-07-02
Version 1.1.3 release of bilby

### Added
- Added `Categorical` prior (!982)(!990)
- Added a built-in mcmc sampler (`bilby_mcmc`) (!905)(!985)
- Added run statistics to the `dynesty` meta data (!969)
- Added `cdf` method to `PriorDict` classes (!943)

### Changes
- Removed the autoburnin causing `kombine` to fail the CI tests (!988)
- Sped up the spline interpolation in ROQ (!971)
- Replaced bessel interpolant to scipy function (!976)
- Improved checkpoint stats plot (!977)
- Fixed a typo in the sampler documentation (!986)
- Fixed issue that causes ConditionalDeltaFunction posterior samples not to be saved correctly (!973)
- Solved an issue where injected SNRs were logged incorrectly (!980)
- Made Python 3.6+ a specific requirement (!978)
- Fixed the calibration and time marginalized likelihood (!978)
- Removed a possible error in the distance marginalization (!960)
- Fixed an issue where `check_draw` did not catch `np.nan` values (!965)
- Removed a superfluous line in the docs configuration file (!963)
- Added a warning about class side effects to the `GravtiationalWaveTransient` likelihood classes (!964)
- Allow `ptemcee` initialization with array (!955)
- Removed `Prior.test_valid_for_rescaling` (!956)
- Replaced deprecated numpy aliases builtins (!970)
- Fixed a bug in the algorithm to determine time resolution of ROQ (!967)
- Restructured utils module into several submodules. API remains backwards compatible (!873)
- Changed number of default walks in `dynesty` from `10*self.ndim` to `100` (!961)

## [1.1.2] - 2021-05-05
Version 1.1.2 release of bilby

### Added
- Added MCMC combine method and improved shuffle behaviour when combining results (!945)
- Added an extras requires to enable downstream packages to depend on `bilby.gw` (!939)
- Added a dynesty unit plot (!954)

### Changes
- Removed a number of deprecated functions and classes (!936)
- Removed the pin on the numpy version (!934)
- Added requirements to MANIFEST (!929)
- Sped up the ROQ weight calculation with IFFT (!903)
- Streamlined hdf5 improvements (!925)
- Sped up `import bilby` by reducing internal imports (!933)
- Reduced time required for the sampler tests (!949)
- Resolved an unclear error message (!935)
- Encapsulated GMST method in `gw.utils` (!947)
- Improvements to `gw.utils` (!948)
- Improvements to `core.prior` (!944)
- Suppresses error message when creating injections (!938)
- Fixed loading meta data, booleans, string lists with hdf5 (!941)
- Made tables an optional requirement (!930)
- Added `exists_ok` to `mkdir` calls (!946)
- Increased the default dynesty checkpoint time to 30 minutes (!940)
- Resolved issue with sampling from prior test (!950)
- Added errstate ignore to the gw.conversion module (!952)
- Fixed issues with pickle saving and loading (!932)
- Fixed an issue with the `_base_roq_waveform` (!959)

## [1.1.1] - 2021-03-16
Version 1.1.1 release of bilby

### Changes
- Added `include requirements.txt` in `MANIFEST.in` to stop the pip installation from breaking

## [1.1.0] - 2021-03-15
Version 1.1.0 release of bilby

### Added
- Calibration marginalisation using a discrete set of realisations (!856)
- Nessai sampler (!921, !926)
- Capability to sample in aligned spin and spin magnitude (!868)
- Information gain now stored in the result (!907)
- Added option to save result/interferometers as pickle (!925)
- Added functionality to notch data (!898)
- Added LIGO India Aundha (A1) coordinates (!886)

### Changes
- Fixed periodic keys not working when constrained priors are present in pymultinest (!927)
- Some changes to reweighting likelihoods (!851)
- `CBCPriorDict` is now a `ConditionalPriorDict` (!868)
- Fixed hyper PE example (!910)
- Pinned numpy and pandas version number (!916)
- Fixed an issue with GPS times in `cpnest`
- `deepdish` is now longer a requirement since it lost its support (!925)
- Removed annoying warning message due to use of `newcommand` in latex (!924)
- Interpolation should be slightly faster now because we now access interpolation libraries more directly (!917, !923)
- Documentation now builds properly (!915)
- Fixed a bug caused by `loaded_modules_dict` (!920)
- `_ref_dist` is an attribute now which speeds up distance marginalised runs slightly (!913)
- Cache normalisation for `PriorDict` objects without `Constraint` priors (!914)
- Removed some deprecated `__future__` imports (!911)
- Fixed the behaviour of `plot_waveform_posterior` to use representative samples (!894)
- Uses `tqdm.auto` in some samplers now for better progress bars (!895)
- Fixed the correction of the epoch in time domain waveforms when using a segment duration that is not a power of two (!909)
- Fixed `ultranest` from failing
- Fixed issues with plotting failing in tests (!904)
- Changed the CI to run on auto-built images (!899)
- Resolved a `matplotlib` error occurring at `dynesty` checkpoint plots (!902)
- Fixed the multidimensional Gaussian example (!901)
- Now allow any lal dictionary option and added a numerical relativity file (!896)
- Fixed the likelihood count in `dynesty` (!853)
- Changed the ordering of keyword arguments for the `Sine` and `Cosine` constructors (!892)

## [1.0.4] - 2020-11-23
Version 1.0.4 release of bilby

### Added
- Added a chirp-mass and mass-ratio prior which are uniform in component masses (!891)

### Changes
- Fixed issue in the CI

## [1.0.3] - 2020-10-23

Version 1.0.3 release of bilby

### Added
- SlabSpikePrior and examples (!857)
- Authors file (!885)
- CDF function to conditional priors (!882)
- Waveform plot in visualising_the_results.ipynb (!817)
- Addition of dnest4 sampler (!849, !883)
- Loaded modules added to meta-data (!881)

### Changes
- Constraint to Uniform priors in ROQ tutorial (!884)
- Fix to CDF and PDF for SymmetricLogUniform prior (!876)
- Fix bug in evidence combination (!880)
- Typo fixes (!878, !887, !879)
- Minor bug fixes (!888)

## [1.0.2] - 2020-09-14

Version 1.0.2 release of bilby

### Added
- Template for the docker files (!783)
- New delta_phase parameter (!850)
- Normalization factor to time-domain waveform plot (!867)
- JSON encoding for int and float types (!866)
- Various minor formatting additions (!870)

### Changes
- Switched to the conda-forge version of multinest and ultranest (!783)
- Updates KAGRA - K1 interferometer information (!861)
- Restructures to tests to be uniform across project (!834)
- Fix to distance and phase marginalization method (!875)
- Fixed roundoff of in-plane spins samples with vectorisation (!864)
- Fix to reference distance and interpolant behavior (!858)
- Fix to constraint prior sampling method (!863)
- Clean up of code (!854)
- Various minor bug, test and plotting fixes (!859, !874, !872, !865)

## [1.0.1] - 2020-08-29

Version 1.0.1 release of bilby

### Added
- Added an rcparams configuration for plotting (!832)
- Added `chi_1` and `chi_2` parameters to default latex label dictionary (!841)
- Allow output merged result file to be gzip or saved as a HDF5 file (!802)

### Changes
- Fixed first value in EOS cumulative integral(!860)
- Fixed saving the number of likelihood evaluations (!848)
- Likelihood condition is now strictly increasing (!846)
- Fixed a minor issue with conditional priors that could cause unexpected behaviour in edge cases (!838)
- Fixed `__repr__` method in the `FromFile` prior (!836)
- Fixed an issue that caused problems for some users when plotting with a latex backend (!816)
- Fixed bug that occurred when min/max of interpolated priors was changed (!815)
- Fixed time domain waveform epoch (!736)
- Fixed time keeping in multinest (!830)
- Now checks if marginalised priors were defined before marginalising (!829)
- Fixed an issue with multivariate Gaussian prior (!822)
- Various minor code improvements (!836)(!839)
- Various minor bug fixes and improvements to the documentation (!820)(!823)(!837)
- Various testing improvements (!833)(!847)(!855)(!852)

## [1.0.0] - 2020-07-06

Version 1.0 release of bilby

### Changes
- Minor bug fixes and typo changes only from 0.6.9, see the [1.0.0 milestone](https://github.com/bilby-dev/bilby/-/merge_requests?scope=all&state=all&milestone_title=1.0.0) for details

## [0.6.9] 2020-05-21
### Changes
- Improvement to the proposal step in dynesty (!774)
- Fix a bug in checking and making directories (!792)
- Clean up of the default prior files (!789)

## [0.6.8] 2020-05-13
### Added
- Option to sample in the sky frame (!786)
- Multiprocessing to reconstruction of marginalized parameters (!782)
- Generic reweighting method for likelihood / priors (!776)
- Parameterized EOS sampling (!543)
- Implementation of the UltraNest sampler (!766)
- Implementation of arVix result files (!772)
- Added basic pre-commit behaviour (!763)

### Changes
- Updated the default PSD to O4 (!757)
- Make multinest allow long file names, optional and work with MPI (!764 !785)
- Add min/max to aligned spin prior (!787)
- Reduce redundant code (!703)
- Added testing for python 3.8 (!762)
- Improvements to the waveform plot (!769)

## [0.6.7] 2020-04-15
### Changes
- Allow dynesty to run with multiprocessing (!754)
- Rewrite ptemcee implementation (!750)
- Change 'source frame' to 'detector frame' in L34-35 of compare_samplers tutorial (!745)
- Allow lal dictionary to be passed through to '_base_lal_cbc_fd_waveform' (!752)

## [0.6.6] 2020-03-06
### Changes
- Fix bug where injected values are not present for corner plot (!749)
- Significant backwards-incompatible improvements to `dynesty` checkpointing (!746)
- Improve checkpoint interval calculation with `dynesty` (!741)
- Fix reading of `PriorDict` class from result file (!739)
- Fix definition of time for time-domain `lalsimulation` waveforms (!736)
- LaTeX text formatting for plots by default (!702)

### Added
- Normalisation dynamically computed when using prior constraints (!704)

## [0.6.5] 2020-02-14
### Changes
- Fix for time reconstruction bug (!714)
- Resolved errors Waveform longer than the frequency array (!710)
- Prior reading clean-up (!715)
- More efficient dynesty restarting (!713)
- PP tests show 123 sigma bounds by default (!726)

### Added
- HealPixPrior (!651)
- GW prior documentation (!720)
- Multiple contours to PP tests plots (!721) 
- Distance marginalization for non-luminosity-distance parameters (!719)

### Removed
- Pipenv (!724)


## [0.6.4] 2020-01-30
### Changes
- Discontinue python2.7 support (!697)
- Minor adjustments to the act calculation method (!679, !707)
- Restructure of the prior module (!688)
- Improvements to the documentation (!708, !700)
- Bug fix when maximum < minimum (!696)

### Added
- Improved waveform error handling (!653)
- Waveform check to the CI (!698)

## [0.6.3] 2020-01-03
### Changed
- Fixed an issue with the ROQ segment scaling (!690)

## [0.6.2] 2019-12-20
### Added
- Introduced conditional prior sets (!332)(!673)(!674)
- Introduced joint priors (!668)
- Added a check to make sure sampling time exists before trying to update (!672)
### Changed
- Fixed a caching issue with the waveform generators (!630)
- Fixed an issue that made the dynamic dynesty sampler not work (!667)
- Changed the backend docker files (!669)
- Fixed an error when plotting time domain data when using `filtfilt=True`
- `Interped` priors now dynamically update when new `yy` values are set (!675)
- Fixed the ROQ scaling checks (!678)

## [0.6.1] 2019-12-02

HotFix release following 0.6.0 fixing a minor bug in the generation of derived
parameters.

## [0.6.0] 2019-12-02
### Added
- A bilby-implemenatation of the dynesty rwalk proposal method (!640)
- An ACT estimate to the rwalk option (!643)
- HTML waveform plots and general improvements to the waveform plot (!641, !659)
- Add a function to resample bilby generated with a uniform in mass ratio and
  chirp_mass prior to uniform in component mass (!642)
- Adds checking to the ROQ usage: warning messages generated when out of bounds (!549)
- A safety check to the time reconstrucion (!633)
- Added the kombine sampler (!637)
- Added in-plane spins (!646)
## Changes
- Changed the PriorDict base class to from OrderedDict to just dict. This fixes
  an issue with pickling priors (!652)
- Improvements to the behaviour of the conversion functions (!647)
- Prevent chirp mass railing for the GW150914 examples (!635)


## [0.5.9] 2019-10-25

### Added
- Default reflective boundaries for calibration parameters !623
- Support for inferring method arguments !608

## Changes
- Speed up the prior evaluations by implementing directly with checks to scipy !627
- Soft initialisation option for the Sampler class !620
- Improvements to JSON reading and writing for functions !621
- Fixed bug in prior reading !618 !617
- Fixes to the examples !619 !614 !626 !616
- Update to the test mode storing extra information !613
- Minor improvements to the ptemcee sampler
- Improved contributing guidelines !610

## Removed
- Default printing of bilby version at import !608

## [0.5.8] 2019-09-26

### Added
- Progress bar in post-processing step

### Changed
- Fixed implementation of calibration !607
- Fixed interaction with dynesty for reflective bounds !604
- Fixed behaviour of n_effective with check pointing !603
- Fixed testing of constants !605
- Fixed bug in bilby_result with python2.7

## [0.5.7] 2019-09-19

### Added
- bilby_convert_resume file CL tool for converting dynesty resume files into preresults !599

### Changes
- Change the constants (Msun, REarth etc) to match the values in LAL !597
- Change the Greenwhich Mean Sidereal Time conversion to match the method in LAL !597
- Update dynesty requirement to 1.0.0
- Improve integration of bounds with dynesty !589
- Fixed issue with mutable default argument !596
- Allow the use of n_effective in dynesty !592
- Allow the use of n_periodic in cpnest !591
- Fix bug in dt calc

## [0.5.6] 2019-09-04

### Changes
- Deprecation of the old helper functions (e.g., fetch open data)
- Improvements to the documentation
- Fix a bug in the dt calculations of the GW likelihood
- Various small bug fixes

### Added
- LAL version information in the meta data

## [0.5.5] 2019-08-22

### Added
- Reading/writing of the prior in a JSON format
- Checks for marginalization flags

### Changes
- Improvements to the examples: reorganisation and fixing bugs
- Fixed bug with scipy>=1.3.0 and spline
- Removed the sqrt(2) normalisation from the scalar longitudinal mode
- Improve PSD filename reading (no longer required "/" to read local files)
- Fix bug in emcee chains
- Added a try/except clause for building the lookup table

## [0.5.4] 2019-07-30

### Added
- Analytic CDFs 
- Reading/writing of grid results objects

### Changed
- Dynesty default settings changed: by default, now uses 30xndim walks. This was
shown (!564) to provide better convergence for the long-duration high-spin tests.
- Fix bug in combined runs log evidence calculations
- Fixed bugs in the nightly tests 

## [0.5.3] 2019-07-23
### Added
- Jitter time marginalization. For the time-marginalized likelihood, a jitter
  is used to ensure proper sampling without artifacts (!534)
- Zero likelihood mode for testing and zero-likelihood test to the nightly C.I ((!542)
- 15D analytic Gaussian test example (!547) 

### Changes
- Dynesty version minimum set to 0.9.7. Changes to this sampler vastly improve
  performance (!537)
- Improvements to waveform plotting (!534) 
- Fixed bugs in the prior loading and added tests (!531 !539 !553 !515)
- Fixed issue in 1D CDF prior plots (!538)
- ROQ weights stored as npz rather than json (memory-performance improvement) (!536)
- Distance marginalisation now uses cubic rather than linear interpolation. Improves
  distance/inclination posteriors for high SNR systems. (!552)
- Inputs to hyperpe modified to allow for more flexible sampling prior specification
  and improve efficiency. (!545)
- Fix definition of some spin phase parameters (!556).

## [0.5.2] 2019-06-18
### Added
- Method to read data in using gwpy get (and associated example)
- Adds a catch for broken resume files with improves reporting

### Changed
- Updated and fixed bugs in examples
- Resolve sampling time persistence for runs which are interrupted
- Improvements to the PP plot
- Speed up of the distance calculation
- Fixed a bug in the inteference of bilby command line arguments with user specified command lines
- Generalised the consistency checks for ResultLists
- Fixes to some tests
- Makes the parameter conversion a static method rather than a lambda expression

## [0.5.1] 2019-06-05
### Added
- Option for the GraceDB service URL
- Precessing BNS
- Functionality to make a waveform plot

### Changed
- Changes to ROQ weight generation: finer time-steps and fixed a bug in the time definition
- Fixed typo "CompactBinaryCoalesnce" -> "CompactBinaryCoalescence" (old class now has deprecation warning)
- Fixed a minor bug in the frequency mask caching
- Minor refractoring of the GWT likelihood and detector tests
- Initial samples in dynesty now generated from the constrained prior

## [0.5.0] 2019-05-08

### Added
- A plot_skymap method to the CBCResult object based on ligo.skymap
- A plot_calibration_posterior method to the CBCResult object
- Method to merge results

### Changed
- Significant refactoring of detector module: this should be backward conmpatible. This work was done to break the large detector.py file into smaller, more manageable chunks. 
- The `periodic_boundary` option to the prior classes has been changed to `boundary`.
*This breaks backward compatibility*.
The options to `boundary` are `{'periodic', 'reflective', None}`.
Periodic boundaries are supported as before.
Reflective boundaries are supported in `dynesty` and `cpnest`.  
- Minor speed improvements by caching intermediate steps
- Added state plotting for dynesty. Use `check_point_plot=True` in the `run_sampler` 
function to create trace plots during the dynesty checkpoints
- Dynesty now prints the progress to STDOUT rather than STDERR
- `detector` module refactored into subpackage. Maintains backward compatibility.
- Specifying alternative frequency bounds for the ROQ now possible if the appropriate
`params.dat` file is passed.

### Removed
- Obsolete (and potentially incorrect) plot_skymap methods from gw.utils

## [0.4.5] 2019-04-03

### Added
- Calibration method and plotting
- Multivariate Gaussian prior
- Bayesian model diminsionality calculator
- Dynamic dynesty (note: this is in an alpha stage)
- Waveform caching

### Changes
- Fixed bugs in the ROQ time resolution
- Fixed bugs in the gracedb wrapper-method
- Improvements to the pp-plot method
- Improved checkpointing for emcee/ptemcee
- Various perforance-related improvements

## [0.4.4] 2019-04-03

### Added
- Infrastucture for custom jump proposals (cpnest-only)
- Evidence uncertainty estimate to cpnest

### Changed
- Bug fix to close figures after creation
- Improved the frequency-mask to entirely remove values outside the mask rather
  than simply set them to zero
- Fix problem with Prior prob and ln_prob if passing multiple samples
- Improved cpnest prior sampling

### Removed
-

## [0.4.3] 2019-03-21

### Added
- Constraint prior: in prior files you can now add option of a constraint based
on other parameters. Currently implements mass-constraints only.
- Grid likelihood: module to evaluate the likelihood on a grid

### Changed
- The GWTransientLikelihood no longer returns -inf for  m2 > m1. It will evaluate
the likelihood as-is. To implement the constraint, use the Constraint priors.

## [0.4.2] 2019-03-21

### Added
- Fermi-Dirac and SymmetricLogUniform prior distributions
- Multivariate Gaussian example and BNS example
- Added standard GWOSC channel names
- Initial work on a fake sampler for testing
- Option for aligned spins
- Results file command line interface
- Full reconstruction of marginalized parameters

### Changed
- Fixed scheduled tests and simplify testing environment
- JSON result files can now be gzipped
- Reduced ROQ memory usage
- Default checkpointing in cpnest

## [0.4.1] 2019-03-04

### Added
- Support for JSON result files
- Before sampling a test is performed for redundant priors

### Changed
- Fixed the definition of iota to theta_jn. WARNING: this breaks backward compatibility. Previously, the CBC parameter iota was used in prior files, but was ill-defined. This fixes that, requiring all scripts to use `theta_jn` in place of `iota`
- Changed the default result file store to JSON rather than hdf5. Reading/writing of hdf5 files is still intact. The read_in_result function will still read in hdf5 files for backward compatibility
- Minor fixes to the way PSDs are calculated
- Fixed a bug in the CBC result where the frequency_domain model was pickled
- Use pickling to store the dynesty resume file and add a write-to-resume on SIGINT/SIGKILL
- Bug fix in ROQ likelihood
- Distance and phase marginalisation work with ROQ likelihood
- Cpnest now creates checkpoints (resume files) by default

### Removed
-

## [0.4.0] 2019-02-15

### Changed
- Changed the logic around redundancy tests in the `PriorDict` classes
- Fixed an accidental addition of astropy as a first-class dependency and added a check for missing dependencies to the C.I.
- Fixed a bug in the "create-your-own-time-domain-model" example
- Added citation guide to the readme

## [0.3.6] 2019-02-10

### Added
- Added the PolyChord sampler, which can be accessed by using `sampler='pypolychord'` in `run_sampler`
- `emcee` now writes all progress to disk and can resume from a previous run.

### Changed
- Cosmology generalised, users can now specify the cosmology used, default is astropy Planck15
- UniformComovingVolume prior *requires* the name to be one of "luminosity_distance", "comoving_distance", "redshift"
- Time/frequency array generation/conversion improved. We now impose `duration` is an integer multiple of
  `sampling_frequency`. Converting back and forth between time/frequency arrays now works for all valid arrays.
- Updates the bilby.core.utils constants to match those of Astropy v3.0.4
- Improve the load_data_from_cache_file method

### Removed
- Removed deprecated `PriorSet` classes. Use `PriorDict` instead.

## [0.3.5] 2019-01-25

### Added
- Reduced Order Quadrature likelihood
- PTMCMCSampler
- CBC result class
- Additional tutorials on using GraceDB and experts guide on running on events in open data

### Changed
- Updated repository information in Dockerfile for PyMultinest

## [0.3.4] 2019-01-10

### Changes
- Renamed the "basic_tutorial.py" example to "fast_tutorial.py" and created a
 "standard_15d_cbc_tutorial.py"
- Renamed "prior" to "priors" in bilby.gw.likelihood.GravtitationalWaveTransient
  for consistency with bilby.core. **WARNING**: This will break scripts which
  use marginalization.
- Added `outdir` kwarg for plotting methods in `bilby.core.result.Result`. This makes plotting
into custom destinations easier.
- Fixed definition of matched_filter_snr, the interferometer method has become `ifo.inner_product`.

### Added
- log-likelihood evaluations for pymultinest
## [0.3.3] 2018-11-08

Changes currently on master, but not under a tag.

- Removed unnecessary arguments (`ra`, `dec`, `geocent_time`, `psi`) from source functions and replaced them with `**kwargs` where appropriate.
- Renamed `PriorSet` to `PriorDict`
- Renamed `BBHPriorSet` to `BBHPriorDict`
- Renamed `BNSPriorSet` to `BNSPriorDict`
- Renamed `CalibrationPriorSet` to `CalibrationPriorDict`
- Added method to result to get injection recovery credible levels
- Added function to generate a pp-plot from many results to core/result.py
- Fixed a bug which caused `Interferometer.detector_tensor` not to update when `latitude`, `longitude`, `xarm_azimuth`, `yarm_azimuth`, `xarm_tilt`, `yarm_tilt` were updated.
- Added implementation of the ROQ likelihood. The basis needs to be specified by the user.
- Extracted time and frequency series behaviour from `WaveformGenerator` and `InterferometerStrainData` and moved it to `series.gw.CoupledTimeAndFrequencySeries`

### Changes
- Switch the ordering the key-word arguments in `result.read_in_result` to put
  `filename` first. This allows users to quickly read in results by filename
- Result object no longer a child of `dict`. Additionally, the list of
  attributes and saved attributes is standardised
- The above changes effect the saving of posteriors. Users can expect that
  opening files made in python 2(3) which where written in 3(2) may no longer
  work. It was felt that the overheads of maintaining cross-version
  compatibility were too much. Note, working in only python 2 or 3, we do not
  expect users to encounter issues.
- Intermediate data products of samples, nested_samples are stored in the h5
- Time marginalised GravitationalWaveTransient works with arbitrary time priors.

## [0.3.1] 2018-11-06

### Changes
- Make BBH/BNS parameter conversion more logical
- Source frame masses/spins included in posterior
- Make filling in posterior with fixed parameters work
- Bug fixes

## [0.3] 2018-11-02

### Added
- Joint-likelihood added
- PyMC3 works with the GravitationalWaveTransient likelihood
- flake8 syntax checking in CI
- Binary neutron star source model
- Allow units to be included in parameter labels
- Add nested samples to dynesty output
- Add more \_\_repr\_\_ methods
- Add ability to plot max likelihood and draws from the posterior
- Document samplers in more detail
- Added the CPNest sampler
- Adds custom titles to corner plots
- Adds plotting of the prior on 1D marginal distributions of corner plots
- Adds a method to plot time-domain GW data
- Added pipenv as a dependency manager
- Hyperparameter estimation now enables the user to provide the single event evidences
- Add nested samples to nestle output
- Prior and child classes now implement the \_\_eq\_\_ magic method for comparisons
- Added default kwargs for each sampler class
- Added NestedSampler and MCSampler helper classes
- Added sampler_requirements.txt file
- Add AlignedSpin gw prior
- Add units to know prior files
- Add pipenv functionality
- Tests run in custom dockerfiles

### Changes
- Fix construct_cbc_derived_parameters
- Autocorrelation calculation moved into parent class
- Fix interpretation of kwargs for dynesty
- PowerSpectralDensity structure modified
- Fixed bug in get_open_data
- Modified how sampling in non-standard parameters is done, the
  `non_standard_sampling_parameter_keys` kwarg has been removed
- .prior files are no longer created. The prior is stored in the result object.
- Removed external_sampler and external_sampler_function attribute from Sampler
- Made conversion of number of livepoint kwargs consistent and streamlined throughout the Nested sampler classes
- Fix label creation in plot_multiple, evidences and repeated plots.
- Changed the way repr works for priors. The repr can now be used to
re-instantiate the Prior in most cases
- Users can now choose to overwrite existing result files, rather than creating
  a .old file.
- Make likelihood values stored in the posterior correct for dynesty and nestle
- pymultinest output now stored in {outdir}/pm_{label}/

### Removed
- Removes the "--detectors" command line argument (not a general CLI requirement)

## [0.2.2] 2018-09-04

### Added
- Add functionality to sample in redshift and reconstruction of source frame masses.
- Add functionality to combine result objects.
- Enable initial values for emcee to be specified.
- Add support for eccentric BBH

### Changed
- Specifying detectors by name from the default command line options has been removed.
- The prior on polarisation phase has been reduced to [0, pi].
- More prior distributions added.
- More samplers supported, pymc3
- More core likelihoods, Poisson, Student's-t
- Result print function fixed
- Add snr functions as methods of `Interferometer`
- The paths between imports where changed so that calls such as
  `bilby.WaveformGenerator` no longer work. Instead, we need to use
  `bilby.gw.WaveformGenerator`. This was done to keep things cleaner going
  forward (when, for example, there may be multiple wfg's).
- Samplers reorganised into individual files.

## [0.2.1] 2018-07-18

### Added
- InterferometerStrainData now handles both time-domain and frequencu-domain data
- Adds documentation on setting data (https://bilby-dev.github.io/bilby/transient-gw-data.html)
- Checkpointing for `dynesty`: the sampling will be checkpointed every 10 minutes (approximately) and can be resumed.
- Add functionality to plot multiple results in a corner plot, see `bilby.core.result.plot_multiple()`.
- Likelihood evaluations are now saved along with the posteriors.

### Changed
- Changed to using `setuptools` for installation.
- Clean up of real data handling: all data is now windowed with a 0.4s roll off (unless set otherwise) and low-pass filtered.
- Add explicit method to create a power spectral density from time-domain data
- Clean up of `PowerSpectralDensity()` - adds `set_from` methods to handle various ways to define the PSD.
- Clean up of `detectors.py`: adds an `InterferometerStrainData` to handle strain data and `InterferometerSet` to handle multiple interferometers. All data setting should primarily be done through the `Interferometer.set_strain_data..` methods.
- Fix the comments and units of `nfft` and `infft` and general improvement to documentation of data.
- Fixed a bug in create_time_series
- Enable marginalisation over calibration uncertainty in Inteferemeter data.
- Fixed the normalisation of the marginalised `GravtitationalWaveTransient` likelihood.
- Fixed a bug in the detector response.

## [0.2.0] 2018-06-17

First `pip` installable version https://pypi.org/project/BILBY/ .

### Added
- Reoriganisation of the directory into `bilby.core` and `bilby.gw`.
- Reading of frame files.
- Major effort to update all docstrings and add some documentation.
- Marginalized likelihoods.
- Examples of searches for gravitational waves from a Supernova and using a sine-Gaussian.
- A `PriorSet` to handle sets of priors and allows reading in from a standardised prior file (see https://bilby-dev.github.io/bilby/prior.html).
- A standardised file for storing detector data.

### Removed
- All chainconsumer dependency as this was causing issues.


[Unreleased]: https://github.com/bilby-dev/bilby/compare/v2.3.0...main
[2.3.0]: https://github.com/bilby-dev/bilby/compare/v2.2.3...v2.3.0
[2.2.3]: https://github.com/bilby-dev/bilby/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/bilby-dev/bilby/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/bilby-dev/bilby/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/bilby-dev/bilby/compare/v2.1.2...v2.2.0
[2.1.2]: https://github.com/bilby-dev/bilby/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/bilby-dev/bilby/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/bilby-dev/bilby/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/bilby-dev/bilby/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/bilby-dev/bilby/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/bilby-dev/bilby/compare/v1.4.1...v2.0.0
[1.4.1]: https://github.com/bilby-dev/bilby/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/bilby-dev/bilby/compare/1.3.0...v1.4.0
[1.3.0]: https://github.com/bilby-dev/bilby/compare/1.2.1...1.3.0
[1.2.1]: https://github.com/bilby-dev/bilby/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/bilby-dev/bilby/compare/1.1.5...1.2.0
[1.1.5]: https://github.com/bilby-dev/bilby/compare/1.1.4...1.1.5
[1.1.4]: https://github.com/bilby-dev/bilby/compare/1.1.2...1.1.4
[1.1.3]: https://github.com/bilby-dev/bilby/compare/1.1.2...1.1.3
[1.1.2]: https://github.com/bilby-dev/bilby/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/bilby-dev/bilby/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/bilby-dev/bilby/compare/1.0.4...1.1.0
[1.0.4]: https://github.com/bilby-dev/bilby/compare/1.0.3...1.0.4
[1.0.3]: https://github.com/bilby-dev/bilby/compare/1.0.2...1.0.3
[1.0.2]: https://github.com/bilby-dev/bilby/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/bilby-dev/bilby/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/bilby-dev/bilby/compare/0.6.9...1.0.0
