# Read nested gzip Monocle objects through serialized slots, without package methods.
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
out<-args[[2L]];if(dir.exists(out)) stop('fresh_export_required');dir.create(out,recursive=TRUE)
object<-readRDS(gzcon(gzfile(args[[1L]],'rb')))
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
write_frame<-function(frame,path) {
  if(any(vapply(frame,function(x) any(as.character(x)=='__VCC_SOURCE_NA__',na.rm=TRUE),logical(1)))) stop('NA_sentinel_collision')
  write.csv(frame,path,row.names=FALSE,na='__VCC_SOURCE_NA__')
}
export_cds<-function(x,directory) {
  dir.create(directory,recursive=TRUE)
  assays<-attr(x,'assays')
  if('ShallowSimpleListAssays' %in% class(assays)) {
    # Older SummarizedExperiment serialization uses a reference-class environment.
    # Read its stored backing field, avoiding package-loading active bindings.
    environment<-attr(assays,'.xData')
    if(!is.environment(environment) || !exists('.->data',envir=environment,inherits=FALSE)) stop('unsupported_shallow_assay_storage')
    assay_data<-get('.->data',envir=environment,inherits=FALSE)
  } else assay_data<-attr(assays,'data')
  counts<-attr(assay_data,'listData')[['counts']]
  if(is.null(counts) || !('dgCMatrix' %in% class(counts))) stop('explicit_dgCMatrix_counts_required')
  dims<-attr(counts,'Dim');axes<-attr(counts,'Dimnames');metadata<-attr(x,'colData');fields<-attr(metadata,'listData')
  if(!identical(axes[[2]],attr(metadata,'rownames'))) stop('CDS_metadata_axis_mismatch')
  if(anyDuplicated(axes[[1]]) || anyDuplicated(axes[[2]])) stop('CDS_duplicate_axis')
  if(!all(vapply(fields,function(a) is.atomic(a) && length(a)==dims[[2]],logical(1)))) stop('unsupported_CDS_metadata_type')
  frame<-as.data.frame(fields,stringsAsFactors=FALSE,check.names=FALSE)
  write_frame(data.frame(source_barcode=axes[[2]],frame,check.names=FALSE),file.path(directory,'cells.csv'))
  write_frame(data.frame(gene=axes[[1]]),file.path(directory,'genes.csv'))
  write_frame(data.frame(field=names(frame),class=vapply(frame,function(a) paste(class(a),collapse='|'),character(1))),file.path(directory,'metadata_types.csv'))
  x<-attr(counts,'x');i<-attr(counts,'i');p<-attr(counts,'p')
  if(length(p)!=dims[[2]]+1 || p[[1]]!=0 || any(diff(p)<0) || tail(p,1)!=length(x) || length(x)!=length(i) || min(i)<0 || max(i)>=dims[[1]]) stop('invalid_CDS_sparse_structure')
  write_frame(data.frame(genes=dims[[1]],cells=dims[[2]],nnz=length(x)),file.path(directory,'shape.csv'))
  finite<-is.finite(x)
  write_frame(data.frame(layer='counts',encoding='dgCMatrix',genes=dims[[1]],cells=dims[[2]],stored_values=length(x),nonfinite=sum(!finite),negative=sum(x<0,na.rm=TRUE),noninteger=sum(x[finite]!=floor(x[finite])),sum=sum(x[finite]),min=min(x[finite]),max=max(x[finite])),file.path(directory,'layers.csv'))
  for(slot in c('p','i','x')) {a<-attr(counts,slot);con<-file(file.path(directory,paste0('counts.',slot,'.bin')),'wb');if(length(a)) for(start in seq.int(1,length(a),by=1000000)) writeBin(a[start:min(start+999999,length(a))],con,size=if(slot=='x') 8L else 4L,endian='little');close(con)}
  writeLines('completed',file.path(directory,'COMPLETE'))
}
if('data.frame' %in% class(object)) {write_frame(as.data.frame(object),file.path(out,'table.csv'));writeLines('source_table',file.path(out,'OBJECT_TYPE'))} else if('cell_data_set' %in% class(object)) {export_cds(object,file.path(out,'CDS'));writeLines('single_CDS',file.path(out,'OBJECT_TYPE'))} else if(is.list(object)) {
  keys<-names(object);if(is.null(keys)) keys<-as.character(seq_along(object))
  write_frame(data.frame(source_list_index=seq_along(object),source_list_key=keys),file.path(out,'list_keys.csv'))
  for(index in seq_along(object)) {if(!('cell_data_set' %in% class(object[[index]]))) stop('non_CDS_list_member');export_cds(object[[index]],file.path(out,paste0('CDS-',index)))}
  writeLines('CDS_list',file.path(out,'OBJECT_TYPE'))
} else stop('unsupported_serialized_object')
