def set_template(args):

    # Set the templates here
    if args.model == 'VDSR' or args.model == 'SRCNN' or args.model == 'LGCNET':
        args.cubic_input = True

    if 'CSARST' in args.model:
        args.test_block = True
        args.n_basic_modules = 3

    if args.dataset == 'RSCNN':
        args.image_size = 400
    elif args.dataset == 'UCMerced':
        if args.scale[0] == 3:
            args.image_size = 255
        else:
            args.image_size = 256