class MVDTCWTSPRunner:

    def __init__(self,
        hdf5_path,
        biorthogonal,
        qshift,
        period,
        subperiod,
        save_load_dir,
        save=False,
        load=False):

        self.hdf5_path = hdf5_path
        self.biorthogonal = biorthogonal
        self.qshift = qshift
        self.period = period
        self.subperiod = subperiod

        self._init_dirs(
            save_load_dir,
            load,
            save)
        self._init_server_stuff()

        self.wavelets = {s : [] for s in self.subjects}

    def run(self):

        if self.load:
            self._load()
        else:
            self._compute()

    def _init_server_stuff(self, load):

        print 'Initializing servers'

        loaders = dles.get_hr_and_acc_all_subjects(
            self.hdf5_path, None, False)
        self.servers = {}

        for (s, dl_list) in loaders.items():
            s = s[-2:]

            # This is to ensure that all subjects have sufficient data
            # Only necessary if we have to recompute wavelets
            try:
                if not self.load:
                    [dl.get_data() for dl in dl_list]

                self.servers[s] = [BS(dl) for dl in dl_list]
            except Exception, e:
                print 'Could not load data for subject', s
                print e
        
        (self.rates, self.names) = unzip(
            [(dl.get_status()['hertz'], dl.name())
             for dl in loaders.items()[0][1]])
                    
        self.subjects = self.servers.keys()
        server_np = lambda ds, r: ds.rows() / (r * self.period)
        subject_np = lambda s: min(
            [server_np(ds, r) 
             for (ds, r) in zip(self.servers[s], self.rates)])
        self.num_periods = {s : int(subject_np(s))
                            for s in self.subjects}
        self.num_views = len(self.servers.items()[0][1])

    def _init_dirs(self,
        save_load_dir,
        load,
        save):

        print 'Initializing directories'

        self.save_load_dir = save_load_dir 
        self.load= load
        self.save= save

        if save and not load:
            if not os.path.isdir(save_load_dir):
                os.mkdir(save_load_dir)

            model_dir = get_ts('_'.join([
                'MVDTCWTSPR',
                'period',
                str(self.period),
                'subperiod',
                str(self.subperiod)]))

            self.save_load_dir = os.path.join(
                save_load_dir,
                model_dir)

            os.mkdir(self.save_load_dir)
        else:
            self.save_load_dir = save_load_dir

        self.wavelet_dir = self._init_dir(
            'wavelets',
            save_wavelets)


    def _compute(self):

        for subject in self.subjects:

            sp_wavelets = self.sp_wavelets[subject]

            for (p, (Yhs, Yss)) in enumerate(sp_wavelets):
                print 'Stuff'

    def _compute(self):

        for subject in self.subjects:

            print 'Computing subperiod wavelet transforms for subject', subject

            (Yls, Yhs) = self._get_sp_wavelet_transforms(subject)

            for period in xrange(self.num_periods[subject]):
                Yhs_period = [view[period] for view in Yhs]
                Yls_period = [view[period] for view in Yls]

                self.sp_wavelets[subject].append(
                    (Yhs_period, Yls_period))

    def _get_sp_wavelet_transforms(self, subject):

        data = [ds.get_data() for ds in self.servers[subject]]
        factors = [int(self.period * r) for r in self.rates]
        sp_factors = [int(self.subperiod * r) for r in self.rates]

        if self.delay is not None:
            data = [view[int(self.delay * r):] 
                    for (r,view) in zip(self.rates, data)]

        thresholds = [int(view.shape[0] * 1.0 / f)
                      for (view, f) in zip(data, factors)]
        Yls = [[] for i in xrange(self.num_views)]
        Yhs = [[] for i in xrange(self.num_views)]
        complete = False
        k = 0

        while not complete:
            exceeded = [(k+1) >= t for t in thresholds]
            complete = any(exceeded)
            current_data = [view[k*f:(k+1)*f]
                            for (f, view) in zip(factors, data)]
            sp_thresholds = [int(view.shape[0] * 1.0 / sp_f)
                             for (view, sp_f) in zip(data, sp_factors)]
            j = 0
            
            # Add a list to each view for the current super period
            for i in xrange(self.num_views):
                Yls[i].append([])
                Yhs[i].append([])

            while not sp_complete: 
                sp_exceeded = [(j+1) >= sp_t for sp_t in sp_thresholds]
                sp_complete = any(sp_exceeded)
                sp_current_data = [view[j*sp_f:(j+1)*sp_f]
                                   for (sp_f, view) in zip(sp_factors, current_data)]

                for (i, view) in enumerate(sp_current_data):
                    (Yl, Yh, _) = dtcwt.oned.dtwavexfm(
                        view, 
                        int(log(view.shape[0], 2)) - 2,
                        self.biorthogonal,
                        self.qshift)
                    
                    # Append the current subperiod's wavelets to the current period
                    Yls[i][-1].append(Yl)
                    Yhs[i][-1].append(Yh)
               
                j += 1

            k += 1

        return (Yls, Yhs)

