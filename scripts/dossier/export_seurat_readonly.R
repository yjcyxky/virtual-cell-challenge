# Base-R-only extraction. Source files are never written and no Seurat methods run.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
source_path <- args[[1L]]; out <- args[[2L]]
if (dir.exists(out)) stop('fresh_export_directory_required')
dir.create(out, recursive=TRUE)
object <- readRDS(source_path)
writeLines(capture.output(infoRDS(source_path)), file.path(out,'rds_serialization.txt'))
writeLines(capture.output(sessionInfo()), file.path(out,'R_session.txt'))
if (!isS4(object) || !('Seurat' %in% class(object))) stop('expected_Seurat_object')
assays <- attr(object,'assays'); meta <- attr(object,'meta.data')
if (!identical(names(assays),'RNA')) stop('unexpected_assays_requires_explicit_review')
assay <- assays[['RNA']]; counts <- attr(assay,'counts')
if (!('dgCMatrix' %in% class(counts))) stop('expected_counts_dgCMatrix')
shape <- attr(counts,'Dim'); axes <- attr(counts,'Dimnames')
if (!identical(axes[[2L]],rownames(meta))) stop('cell_axis_metadata_mismatch')
if (length(axes[[1L]])!=shape[[1L]] || length(axes[[2L]])!=shape[[2L]]) stop('axis_length_mismatch')
if (anyDuplicated(axes[[1L]]) || anyDuplicated(axes[[2L]])) stop('duplicate_source_axis')
if (!all(c('cell_type','pathway','Batch_info','guide','gene') %in% names(meta))) stop('required_source_labels_missing')
for (key in c('cell_type','pathway','Batch_info','guide','gene')) {
  if (anyNA(meta[[key]]) || any(as.character(meta[[key]])=='')) stop(paste('missing_source_label',key))
}
write.csv(data.frame(gene=axes[[1L]]),file.path(out,'genes.csv'),row.names=FALSE,na='__VCC_SOURCE_NA__')
if (any(vapply(meta,function(x) any(as.character(x)=='__VCC_SOURCE_NA__',na.rm=TRUE),logical(1L)))) stop('NA_sentinel_collision')
write.csv(data.frame(source_barcode=rownames(meta),meta,check.names=FALSE),file.path(out,'cells.csv'),row.names=FALSE,na='__VCC_SOURCE_NA__')
write.csv(data.frame(field=names(meta),class=vapply(meta,function(x) paste(class(x),collapse='|'),character(1L))),file.path(out,'metadata_types.csv'),row.names=FALSE)
audit <- list()
for (name in c('counts','data','scale.data')) {
  matrix <- attr(assay,name)
  if (is.null(matrix)) stop(paste('missing_RNA_slot',name))
  if (isS4(matrix)) {
    if (!('dgCMatrix' %in% class(matrix))) stop(paste('unsupported_RNA_slot',name))
    dims <- attr(matrix,'Dim'); values <- attr(matrix,'x'); ptr <- attr(matrix,'p'); indices <- attr(matrix,'i')
    if (length(ptr)!=dims[[2L]]+1L || ptr[[1L]]!=0L || any(diff(ptr)<0L) || tail(ptr,1)!=length(values) || length(indices)!=length(values)) stop('invalid_CSC_structure')
    if (length(indices) && (min(indices)<0L || max(indices)>=dims[[1L]])) stop('CSC_index_out_of_bounds')
    if (name=='counts' || name=='data') {
      if (!identical(attr(matrix,'Dimnames'),axes) || !identical(dims,shape)) stop('RNA_layer_axis_mismatch')
    }
  } else if (is.matrix(matrix)) {dims <- dim(matrix);values <- as.vector(matrix)} else stop('unsupported_RNA_matrix')
  nbad <- 0; nnegative <- 0; nfractional <- 0; total <- 0; minimum <- Inf; maximum <- -Inf
  if (length(values)) for (start in seq.int(1,length(values),by=1000000)) {
    chunk <- values[start:min(start+999999,length(values))]; finite <- is.finite(chunk)
    nbad <- nbad+sum(!finite); nnegative <- nnegative+sum(chunk<0,na.rm=TRUE)
    nfractional <- nfractional+sum(chunk[finite]!=floor(chunk[finite])); total <- total+sum(chunk[finite])
    minimum <- min(minimum,chunk[finite]);maximum <- max(maximum,chunk[finite])
  }
  audit[[name]] <- data.frame(layer=name,encoding=class(matrix)[[1L]],genes=dims[[1L]],cells=dims[[2L]],stored_values=length(values),nonfinite=nbad,negative=nnegative,noninteger=nfractional,sum=total,min=if(is.finite(minimum)) minimum else NA,max=if(is.finite(maximum)) maximum else NA)
  if (name=='counts' && (nbad || nnegative || nfractional)) stop('invalid_count_values')
}
write.csv(do.call(rbind,audit),file.path(out,'layers.csv'),row.names=FALSE)
write.csv(data.frame(genes=shape[[1L]],cells=shape[[2L]],nnz=length(attr(counts,'x'))),file.path(out,'shape.csv'),row.names=FALSE)
for (slot in c('p','i','x')) {
  values <- attr(counts,slot); connection <- file(file.path(out,paste0('counts.',slot,'.bin')),'wb')
  if (length(values)) for (start in seq.int(1,length(values),by=1000000)) writeBin(values[start:min(start+999999,length(values))],connection,size=if(slot=='x') 8L else 4L,endian='little')
  close(connection)
}
writeLines('RNA/counts gene-by-cell CSC p/i/x, little endian; transposed view is cell-by-gene CSR without numerical transformation.',file.path(out,'export_semantics.txt'))
writeLines('completed',file.path(out,'COMPLETE'))
