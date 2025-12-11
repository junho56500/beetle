
data = {
'root_dir': '/projects/',
    'image_size': (256, 256),      # h x w for input images
    'num_cameras': 4,
    'depth_bins': 100,             # discretization planes for lss lift step
    'min_depth': 1.0,              # start of depth range (in meters)
    'max_depth': 200.0,             # end of depth range (in meters)
    'bev_resolution': 0.5,         # meters per pixel in bev (e.g., 0.5m/pixel)
    'roi': [-50.0, 50.0, -50.0, 200.0],  # (min_x, max_x, min_y, max_y in meters)
}

model = {
    'dim_feat': 256,  # Common feature dimension C
    'num_classes': 10,   # color types such as white, green, yellow
    'iou_thres': 0.5,   # IoU threshold for matching ground truth
    
    'encoder': {
        'backbone': 'resnet50',
        'fpn_out_channels': 256,
        'fpn_levels': [3, 4, 5],   # feature levels used from fpn
    },
    'lss': {
        'z_dim': 8,                # height dimension of the bev feature map (vertical aggregation)
        'bev_channels': 256,       # output channel count after lss splat
    },    
    'encoder': {
        'type': 'self_attention',
        'num_layers': 3,
        'num_heads': 8,
    },
    'decoder': {
        'type': 'deformable_attention',
        'num_layers': 6,
        'num_heads': 8,
        'query_dim': 256,
        'num_object_queries': 900, # number of learned object anchors
    },
}
    
train = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'loss_weights': {
        'cls_focal_loss': 2.0,
        'l1_bbox_loss': 5.0,
        'giou_loss': 2.0,
    },
}

test = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'loss_weights': {
        'cls_focal_loss': 2.0,
        'l1_bbox_loss': 5.0,
        'giou_loss': 2.0,
    },
}

eval = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'loss_weights': {
        'cls_focal_loss': 2.0,
        'l1_bbox_loss': 5.0,
        'giou_loss': 2.0,
    },
}