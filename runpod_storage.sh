#!/bin/bash

# RunPod Storage Management for nanochat
# Utilities for backing up and managing nanochat artifacts on RunPod

WORKSPACE_DIR="/workspace"
NANOCHAT_ARTIFACTS="$WORKSPACE_DIR/nanochat_artifacts"
BACKUP_DIR="$WORKSPACE_DIR/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check RunPod storage status
check_storage_status() {
    print_info "RunPod Storage Status Check"
    echo "================================"
    
    # Workspace disk usage
    echo "💾 Workspace Storage:"
    df -h $WORKSPACE_DIR | tail -1 | awk '{print "  Used: " $3 " / " $2 " (" $5 " full)"}'
    
    # Check if nanochat artifacts exist
    if [ -d "$NANOCHAT_ARTIFACTS" ]; then
        ARTIFACTS_SIZE=$(du -sh $NANOCHAT_ARTIFACTS 2>/dev/null | cut -f1)
        print_success "nanochat artifacts found: $ARTIFACTS_SIZE"
        
        # Check for specific components
        echo ""
        echo "📦 Artifact Components:"
        
        if [ -d "$NANOCHAT_ARTIFACTS/tokenizer" ]; then
            TOK_SIZE=$(du -sh $NANOCHAT_ARTIFACTS/tokenizer 2>/dev/null | cut -f1 || echo "N/A")
            echo "  ✓ Tokenizer: $TOK_SIZE"
        fi
        
        if ls $NANOCHAT_ARTIFACTS/base_model_* &>/dev/null; then
            BASE_SIZE=$(du -sh $NANOCHAT_ARTIFACTS/base_model_* 2>/dev/null | head -1 | cut -f1 || echo "N/A")
            echo "  ✓ Base model: $BASE_SIZE"
        fi
        
        if ls $NANOCHAT_ARTIFACTS/mid_model_* &>/dev/null; then
            MID_SIZE=$(du -sh $NANOCHAT_ARTIFACTS/mid_model_* 2>/dev/null | head -1 | cut -f1 || echo "N/A")
            echo "  ✓ Midtrained model: $MID_SIZE"
        fi
        
        if ls $NANOCHAT_ARTIFACTS/sft_model_* &>/dev/null; then
            SFT_SIZE=$(du -sh $NANOCHAT_ARTIFACTS/sft_model_* 2>/dev/null | head -1 | cut -f1 || echo "N/A")
            echo "  ✓ SFT model: $SFT_SIZE"
        fi
        
        if [ -f "$WORKSPACE_DIR/nanochat_final_report.md" ]; then
            echo "  ✓ Final report: Available"
        fi
        
    else
        print_warning "No nanochat artifacts found"
    fi
    
    echo ""
    
    # Check for backups
    if [ -d "$BACKUP_DIR" ]; then
        BACKUP_COUNT=$(ls -1 $BACKUP_DIR/*.tar.gz 2>/dev/null | wc -l)
        if [ $BACKUP_COUNT -gt 0 ]; then
            BACKUP_SIZE=$(du -sh $BACKUP_DIR 2>/dev/null | cut -f1)
            print_success "Found $BACKUP_COUNT backup(s): $BACKUP_SIZE total"
        fi
    fi
}

# Function to create compressed backup
create_backup() {
    local backup_name=${1:-"nanochat_$(date +%Y%m%d_%H%M%S)"}
    
    if [ ! -d "$NANOCHAT_ARTIFACTS" ]; then
        print_error "No nanochat artifacts found to backup"
        return 1
    fi
    
    mkdir -p "$BACKUP_DIR"
    
    print_info "Creating backup: $backup_name.tar.gz"
    
    cd $WORKSPACE_DIR
    tar -czf "$BACKUP_DIR/${backup_name}.tar.gz" \
        nanochat_artifacts/ \
        nanochat_final_report.md 2>/dev/null || true
    
    if [ $? -eq 0 ]; then
        BACKUP_SIZE=$(du -sh "$BACKUP_DIR/${backup_name}.tar.gz" | cut -f1)
        print_success "Backup created: ${backup_name}.tar.gz ($BACKUP_SIZE)"
        return 0
    else
        print_error "Backup creation failed"
        return 1
    fi
}

# Function to restore from backup
restore_backup() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        print_error "Please specify backup file"
        echo "Available backups:"
        ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "No backups found"
        return 1
    fi
    
    if [ ! -f "$BACKUP_DIR/$backup_file" ] && [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Use full path if not in backup dir
    if [ -f "$backup_file" ]; then
        RESTORE_FILE="$backup_file"
    else
        RESTORE_FILE="$BACKUP_DIR/$backup_file"
    fi
    
    print_warning "This will overwrite existing nanochat artifacts"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup existing artifacts if they exist
        if [ -d "$NANOCHAT_ARTIFACTS" ]; then
            EXISTING_BACKUP="${BACKUP_DIR}/pre_restore_$(date +%Y%m%d_%H%M%S).tar.gz"
            print_info "Backing up existing artifacts to: $(basename $EXISTING_BACKUP)"
            cd $WORKSPACE_DIR
            tar -czf "$EXISTING_BACKUP" nanochat_artifacts/ 2>/dev/null
        fi
        
        # Restore from backup
        print_info "Restoring from: $RESTORE_FILE"
        cd $WORKSPACE_DIR
        tar -xzf "$RESTORE_FILE"
        
        if [ $? -eq 0 ]; then
            print_success "Restore completed successfully"
        else
            print_error "Restore failed"
            return 1
        fi
    else
        print_info "Restore cancelled"
    fi
}

# Function to download artifacts to local machine
download_instructions() {
    print_info "Download Instructions for Local Machine"
    echo "======================================"
    echo ""
    echo "To download your nanochat artifacts to your local machine:"
    echo ""
    echo "1. 💾 Create backup first (if not already done):"
    echo "   $0 backup my_nanochat_model"
    echo ""
    echo "2. 📡 Download via RunPod SSH:"
    echo "   scp -P [SSH_PORT] root@[RUNPOD_IP]:/workspace/backups/my_nanochat_model.tar.gz ."
    echo ""
    echo "3. 🌐 Or use RunPod's web file manager:"
    echo "   - Go to RunPod web interface"
    echo "   - Click 'Connect' → 'Start Web Terminal'"
    echo "   - Navigate to /workspace/backups/"
    echo "   - Download the .tar.gz file"
    echo ""
    echo "4. 📤 Upload to cloud storage (optional):"
    echo "   # AWS S3 example:"
    echo "   aws s3 cp /workspace/backups/my_nanochat_model.tar.gz s3://your-bucket/"
    echo ""
    echo "   # Google Cloud Storage example:"
    echo "   gsutil cp /workspace/backups/my_nanochat_model.tar.gz gs://your-bucket/"
    echo ""
    print_warning "Remember: RunPod charges for storage while pod is running"
    print_info "Consider downloading/uploading to external storage for long-term keeping"
}

# Function to setup cloud storage integration
setup_cloud_storage() {
    echo "🌩️  Cloud Storage Setup Options"
    echo "==============================="
    echo ""
    echo "Choose your preferred cloud storage:"
    echo "1) AWS S3"
    echo "2) Google Cloud Storage"
    echo "3) Azure Blob Storage"
    echo "4) Skip cloud setup"
    echo ""
    read -p "Enter choice (1-4): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            print_info "Setting up AWS S3..."
            echo "Run the following commands:"
            echo "  apt update && apt install -y awscli"
            echo "  aws configure  # Enter your AWS credentials"
            echo ""
            echo "Then use: aws s3 cp /workspace/backups/backup.tar.gz s3://your-bucket/"
            ;;
        2)
            print_info "Setting up Google Cloud Storage..."
            echo "Run the following commands:"
            echo "  curl https://sdk.cloud.google.com | bash"
            echo "  exec -l \$SHELL"
            echo "  gcloud init  # Authenticate with your Google account"
            echo ""
            echo "Then use: gsutil cp /workspace/backups/backup.tar.gz gs://your-bucket/"
            ;;
        3)
            print_info "Setting up Azure Blob Storage..."
            echo "Run the following commands:"
            echo "  curl -sL https://aka.ms/InstallAzureCLIDeb | bash"
            echo "  az login  # Authenticate with your Azure account"
            echo ""
            echo "Then use: az storage blob upload --file backup.tar.gz --name backup.tar.gz"
            ;;
        4)
            print_info "Skipping cloud storage setup"
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Function to clean up old artifacts/backups
cleanup() {
    echo "🧹 Cleanup Options"
    echo "=================="
    echo "1) Remove old artifacts (keep latest checkpoint only)"
    echo "2) Remove old backups (keep latest 3)"
    echo "3) Clean temporary files"
    echo "4) Full cleanup (removes everything except latest backup)"
    echo ""
    read -p "Enter choice (1-4): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            print_info "Cleaning old model checkpoints..."
            # Keep only the latest checkpoint for each model type
            find "$NANOCHAT_ARTIFACTS" -name "model_*.pt" -type f -print0 | \
                xargs -0 ls -t | tail -n +2 | xargs rm -f 2>/dev/null || true
            print_success "Old checkpoints removed"
            ;;
        2)
            print_info "Cleaning old backups (keeping latest 3)..."
            if [ -d "$BACKUP_DIR" ]; then
                ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
                print_success "Old backups removed"
            fi
            ;;
        3)
            print_info "Cleaning temporary files..."
            find /tmp -name "*nanochat*" -type f -mtime +1 -delete 2>/dev/null || true
            find "$WORKSPACE_DIR" -name "*.log" -mtime +1 -delete 2>/dev/null || true
            print_success "Temporary files cleaned"
            ;;
        4)
            print_warning "This will remove ALL artifacts except the latest backup!"
            read -p "Are you sure? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Create backup first if none exists
                if [ ! -d "$BACKUP_DIR" ] || [ $(ls -1 "$BACKUP_DIR"/*.tar.gz 2>/dev/null | wc -l) -eq 0 ]; then
                    print_info "Creating backup before cleanup..."
                    create_backup "pre_cleanup_$(date +%Y%m%d_%H%M%S)"
                fi
                
                rm -rf "$NANOCHAT_ARTIFACTS"
                find "$BACKUP_DIR" -name "*.tar.gz" -type f -print0 | \
                    xargs -0 ls -t | tail -n +2 | xargs rm -f 2>/dev/null || true
                
                print_success "Full cleanup completed"
            fi
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Main command dispatcher
case "$1" in
    "status"|"check")
        check_storage_status
        ;;
    "backup")
        create_backup "$2"
        ;;
    "restore")
        restore_backup "$2"
        ;;
    "download")
        download_instructions
        ;;
    "cloud")
        setup_cloud_storage
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "RunPod nanochat Storage Management"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  status                    - Check storage status and artifacts"
        echo "  backup [name]            - Create compressed backup of artifacts"
        echo "  restore <backup_file>    - Restore from backup"
        echo "  download                 - Show download instructions"
        echo "  cloud                    - Setup cloud storage integration"
        echo "  cleanup                  - Clean up old files and artifacts"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 backup my_chatgpt_model"
        echo "  $0 restore my_chatgpt_model.tar.gz"
        echo "  $0 download"
        echo ""
        echo "💡 Tip: RunPod /workspace/ directory persists across sessions"
        echo "    but charges storage costs. Use backups for long-term storage."
        exit 1
        ;;
esac