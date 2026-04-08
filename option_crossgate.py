from option import Options as BaseOptions


class Options(BaseOptions):
    def init(self):
        super().init()
        self.parser.add_argument(
            "--crossgate_attn_dim",
            type=int,
            default=64,
            help="compressed channel width used by lightweight cross-gate attention",
        )
        self.parser.add_argument(
            "--crossgate_num_heads",
            type=int,
            default=4,
            help="number of heads used by lightweight cross-gate attention",
        )
        self.parser.add_argument(
            "--crossgate_window_size",
            type=int,
            default=5,
            help="odd sliding-window size used by lightweight cross-gate attention",
        )
        self.parser.add_argument(
            "--crossgate_gamma_init",
            type=float,
            default=0.1,
            help="initial residual scaling for cross-gate attention guidance",
        )
