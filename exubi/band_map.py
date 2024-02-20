class band_map():
    """Raw data from the ARPES experiment

    Parameters
    ----------
    intensities: 2d object
        Some more
    angles: ndarray
        Theta angle values
    ekin: ndarray
        Kinetic energies detected during experiment

    Attributes
    ----------
    """

    
    def __init__(self, intensities, angles, ekin):
        self.intensities=intensities
        self.angles=angles
        self.ekin=ekin