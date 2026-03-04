# Changelog

## [v0.1.7]
### Fixed
- Resolves crash when reading a continued file.

## [v0.1.6]
### Fixed
- Performance improvement using numpy.frombuffer() to read waveforms.
- Fewer calls to struct.unpack().
- Code cleanup.