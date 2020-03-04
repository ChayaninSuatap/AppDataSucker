
import gc
def my_fit_generator(model, train_data_fn, test_data_fn, epochs, batch_size, callbacks,
    class_weight = None, initial_epoch = 0):

    #call on train start callback

    #train and validate each ep
    for epoch_i in range(epochs):

        #train
        train_gen = train_data_fn()
        batch_i = 0
        while True:
            batch_i += 1
            print('fetching')
            try:
                xs, ys = next(train_gen)
            except:
                break
            print('training on epoch %s batch %s len xs %d' % (epoch_i, batch_i, len(xs)))

            train_results = model.train_on_batch(xs, ys, class_weight = class_weight)
            print(train_results)

            gc.collect()

        #test
        test_gen = test_data_fn()
        batch_i = 0
        while True:
            batch_i += 1
            print('fetching')
            try:
                xs, ys = next(test_gen)
            except:
                break
            print('testing on epoch %s batch %s len xs %d' % (epoch_i, batch_i, len(xs)))

            test_results = model.test_on_batch(xs, ys)
            print(test_results)
    
    print(model.metrics_names)

    
    #call on epoch end callback

    #call on train end
